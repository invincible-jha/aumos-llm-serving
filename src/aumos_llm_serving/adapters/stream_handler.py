"""SSE stream handler adapter.

Manages server-sent event (SSE) streams for token-by-token LLM responses.
Handles concurrent stream lifecycle, buffering, client disconnection,
progress tracking, timeout enforcement, and cancellation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SSE_HEARTBEAT_INTERVAL_SECONDS: float = 15.0
_SSE_DONE_SENTINEL: str = "data: [DONE]\n\n"


@dataclass
class StreamSession:
    """Runtime state for a single SSE stream.

    Attributes:
        stream_id: Unique stream identifier.
        tenant_id: Owning tenant UUID.
        model: Model producing tokens.
        started_at: Monotonic timestamp when streaming began.
        timeout_seconds: Hard timeout for this stream.
        tokens_generated: Running count of tokens yielded to the client.
        is_cancelled: Set to True when the client disconnects or cancels.
        is_done: Set to True when the inference engine signals completion.
        last_token_at: Monotonic timestamp of the last token sent.
    """

    stream_id: uuid.UUID
    tenant_id: uuid.UUID
    model: str
    started_at: float
    timeout_seconds: float
    tokens_generated: int = 0
    is_cancelled: bool = False
    is_done: bool = False
    last_token_at: float = field(default_factory=time.monotonic)


def _format_sse_chunk(data: str) -> str:
    """Wrap a JSON string as a well-formed SSE data line.

    Args:
        data: The JSON payload string.

    Returns:
        SSE-formatted string with trailing double newline.
    """
    return f"data: {data}\n\n"


def _format_sse_error(message: str, stream_id: str) -> str:
    """Format an SSE error event.

    Args:
        message: Human-readable error message.
        stream_id: Stream identifier for client-side correlation.

    Returns:
        SSE event string.
    """
    import json
    payload = json.dumps({"error": message, "stream_id": stream_id})
    return f"event: error\ndata: {payload}\n\n"


def _format_sse_heartbeat() -> str:
    """Format an SSE keep-alive comment line.

    Returns:
        SSE comment string that keeps the HTTP connection alive.
    """
    return ": heartbeat\n\n"


class StreamHandler:
    """SSE token streaming manager for concurrent LLM streams.

    Wraps raw token iterators from inference providers and delivers them to
    HTTP clients via the SSE protocol. Manages per-stream state, enforces
    timeouts, detects client disconnection, and tracks token progress.

    Typical usage in a FastAPI streaming endpoint:

        handler = StreamHandler(default_timeout_seconds=120.0)
        async for chunk in handler.stream(
            source=provider.stream_chat_completion(request),
            tenant_id=tenant_id,
            model=request.model,
        ):
            yield chunk
    """

    def __init__(
        self,
        default_timeout_seconds: float = 120.0,
        buffer_size: int = 64,
        heartbeat_interval_seconds: float = _SSE_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the stream handler.

        Args:
            default_timeout_seconds: Default stream timeout in seconds.
            buffer_size: Internal token buffer size (for back-pressure).
            heartbeat_interval_seconds: Interval between SSE keep-alive comments.
        """
        self._default_timeout_seconds = default_timeout_seconds
        self._buffer_size = buffer_size
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

        # Active streams registry
        self._sessions: dict[uuid.UUID, StreamSession] = {}

    async def stream(
        self,
        source: AsyncIterator[str],
        tenant_id: uuid.UUID,
        model: str,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream SSE token chunks from a provider source iterator.

        Wraps the raw provider stream with timeout enforcement, client
        disconnection detection, heartbeat injection, and progress tracking.

        Args:
            source: Async iterator yielding raw SSE chunks from the provider.
            tenant_id: Owning tenant for logging and metrics.
            model: Model producing the stream.
            timeout_seconds: Per-stream timeout; defaults to instance default.

        Yields:
            Formatted SSE data chunks suitable for HTTP streaming responses.

        Raises:
            asyncio.TimeoutError: If the stream exceeds the timeout.
        """
        effective_timeout = timeout_seconds or self._default_timeout_seconds
        stream_id = uuid.uuid4()
        session = StreamSession(
            stream_id=stream_id,
            tenant_id=tenant_id,
            model=model,
            started_at=time.monotonic(),
            timeout_seconds=effective_timeout,
        )
        self._sessions[stream_id] = session

        logger.info(
            "SSE stream started",
            stream_id=str(stream_id),
            tenant_id=str(tenant_id),
            model=model,
            timeout_seconds=effective_timeout,
        )

        try:
            async for chunk in self._guarded_stream(source, session):
                yield chunk
        finally:
            duration_ms = int((time.monotonic() - session.started_at) * 1000)
            logger.info(
                "SSE stream ended",
                stream_id=str(stream_id),
                tenant_id=str(tenant_id),
                model=model,
                tokens_generated=session.tokens_generated,
                duration_ms=duration_ms,
                cancelled=session.is_cancelled,
            )
            self._sessions.pop(stream_id, None)

    async def _guarded_stream(
        self,
        source: AsyncIterator[str],
        session: StreamSession,
    ) -> AsyncIterator[str]:
        """Apply timeout and heartbeat logic around the raw source iterator.

        Args:
            source: Raw provider SSE chunk iterator.
            session: Active StreamSession tracking state.

        Yields:
            Processed SSE chunks with heartbeats injected.
        """
        last_heartbeat_at = time.monotonic()

        while not session.is_cancelled:
            # Enforce per-stream wall-clock timeout
            elapsed = time.monotonic() - session.started_at
            if elapsed >= session.timeout_seconds:
                logger.warning(
                    "SSE stream timed out",
                    stream_id=str(session.stream_id),
                    elapsed_seconds=round(elapsed, 1),
                    timeout_seconds=session.timeout_seconds,
                )
                yield _format_sse_error(
                    message=f"Stream timeout after {session.timeout_seconds}s",
                    stream_id=str(session.stream_id),
                )
                return

            # Inject heartbeat if needed to prevent proxy/load-balancer timeout
            now = time.monotonic()
            if now - last_heartbeat_at >= self._heartbeat_interval_seconds:
                yield _format_sse_heartbeat()
                last_heartbeat_at = now

            # Try to get next token with a short poll window
            try:
                remaining = session.timeout_seconds - elapsed
                chunk = await asyncio.wait_for(
                    self._next_chunk(source),
                    timeout=min(1.0, remaining),
                )
            except asyncio.TimeoutError:
                # No token available yet; loop back to check heartbeat/timeout
                continue
            except StopAsyncIteration:
                # Inference complete
                session.is_done = True
                yield _SSE_DONE_SENTINEL
                return
            except Exception as exc:
                logger.error(
                    "SSE stream source error",
                    stream_id=str(session.stream_id),
                    error=str(exc),
                )
                yield _format_sse_error(
                    message=f"Inference error: {exc}",
                    stream_id=str(session.stream_id),
                )
                return

            if chunk == _SSE_DONE_SENTINEL or chunk.strip() == "data: [DONE]":
                session.is_done = True
                yield _SSE_DONE_SENTINEL
                return

            # Count tokens from the chunk (approximate: one chunk ≈ one token)
            session.tokens_generated += 1
            session.last_token_at = time.monotonic()
            yield chunk

    @staticmethod
    async def _next_chunk(source: AsyncIterator[str]) -> str:
        """Await the next chunk from the async source iterator.

        Args:
            source: The async iterator to pull from.

        Returns:
            Next string chunk.

        Raises:
            StopAsyncIteration: When the source is exhausted.
        """
        return await source.__anext__()

    async def cancel_stream(self, stream_id: uuid.UUID) -> bool:
        """Cancel an active SSE stream.

        Marks the stream as cancelled so the _guarded_stream loop exits
        on its next iteration.

        Args:
            stream_id: ID of the stream to cancel.

        Returns:
            True if the stream was found and cancelled, False otherwise.
        """
        session = self._sessions.get(stream_id)
        if session is None:
            return False
        session.is_cancelled = True
        logger.info(
            "SSE stream cancelled",
            stream_id=str(stream_id),
            tokens_generated_before_cancel=session.tokens_generated,
        )
        return True

    def get_stream_progress(self, stream_id: uuid.UUID) -> dict[str, object] | None:
        """Get current progress of an active stream.

        Args:
            stream_id: Stream to query.

        Returns:
            Dict with tokens_generated, elapsed_seconds, and is_done,
            or None if the stream is not active.
        """
        session = self._sessions.get(stream_id)
        if session is None:
            return None
        return {
            "stream_id": str(stream_id),
            "tenant_id": str(session.tenant_id),
            "model": session.model,
            "tokens_generated": session.tokens_generated,
            "elapsed_seconds": round(time.monotonic() - session.started_at, 2),
            "timeout_seconds": session.timeout_seconds,
            "is_done": session.is_done,
            "is_cancelled": session.is_cancelled,
        }

    def list_active_streams(self) -> list[dict[str, object]]:
        """List all currently active SSE streams.

        Returns:
            List of stream progress dicts for all active sessions.
        """
        return [
            {
                "stream_id": str(session.stream_id),
                "tenant_id": str(session.tenant_id),
                "model": session.model,
                "tokens_generated": session.tokens_generated,
                "elapsed_seconds": round(time.monotonic() - session.started_at, 2),
                "is_done": session.is_done,
                "is_cancelled": session.is_cancelled,
            }
            for session in self._sessions.values()
        ]

    @property
    def active_stream_count(self) -> int:
        """Return the number of currently active streams."""
        return len(self._sessions)
