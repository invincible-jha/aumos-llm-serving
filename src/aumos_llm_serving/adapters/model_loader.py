"""Model loader adapter.

Handles multi-format model discovery, loading, caching, warm-up, and
unloading for self-hosted model deployments. Supports Safetensors, GGUF,
GPTQ, and AWQ formats with LRU-style VRAM-aware caching.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ModelFormat(str, enum.Enum):
    """Supported on-disk model formats."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"
    PYTORCH_BIN = "pytorch_bin"
    UNKNOWN = "unknown"


class ModelStatus(str, enum.Enum):
    """Runtime lifecycle state of a loaded model."""

    LOADING = "loading"
    READY = "ready"
    WARMING_UP = "warming_up"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelEntry:
    """Metadata and handle for a model loaded into memory.

    Attributes:
        model_id: Canonical model identifier.
        model_format: On-disk serialization format.
        vram_bytes: Estimated VRAM usage in bytes.
        status: Current lifecycle status.
        loaded_at: Unix timestamp when loading completed.
        last_used_at: Unix timestamp of last inference request.
        warmup_tokens_generated: Tokens generated during warm-up.
        error_message: Error detail if status is ERROR.
        handle: Raw model object (framework-specific, opaque to this module).
    """

    model_id: str
    model_format: ModelFormat
    vram_bytes: int
    status: ModelStatus
    loaded_at: float
    last_used_at: float
    warmup_tokens_generated: int = 0
    error_message: str | None = None
    handle: Any = field(default=None, repr=False)


def _detect_format(model_path: Path) -> ModelFormat:
    """Detect the serialization format of a model directory or file.

    Args:
        model_path: Path to the model directory or single-file model.

    Returns:
        Detected ModelFormat enum value.
    """
    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix == ".gguf":
            return ModelFormat.GGUF
        if suffix in {".safetensors"}:
            return ModelFormat.SAFETENSORS
    elif model_path.is_dir():
        files = list(model_path.iterdir())
        extensions = {f.suffix.lower() for f in files}
        if ".safetensors" in extensions:
            return ModelFormat.SAFETENSORS
        if ".gguf" in extensions:
            return ModelFormat.GGUF
        # GPTQ detection: look for quantize_config.json with quant_type gptq
        config_path = model_path / "quantize_config.json"
        if config_path.exists():
            try:
                import json
                config = json.loads(config_path.read_text())
                quant_type: str = config.get("quant_type", "").lower()
                if quant_type == "gptq":
                    return ModelFormat.GPTQ
                if quant_type == "awq":
                    return ModelFormat.AWQ
            except Exception:
                pass
        # AWQ: adapter_config.json with awq in quant_type
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                import json
                config = json.loads(adapter_config.read_text())
                if "awq" in str(config).lower():
                    return ModelFormat.AWQ
            except Exception:
                pass
        if ".bin" in extensions:
            return ModelFormat.PYTORCH_BIN
    return ModelFormat.UNKNOWN


def _estimate_vram_bytes(model_path: Path, model_format: ModelFormat) -> int:
    """Estimate VRAM consumption for a model based on file sizes.

    Uses a format-specific multiplier over the raw file size since quantized
    models have different runtime vs. disk size characteristics.

    Args:
        model_path: Path to the model directory or file.
        model_format: Detected format to select the multiplier.

    Returns:
        Estimated VRAM usage in bytes.
    """
    multipliers: dict[ModelFormat, float] = {
        ModelFormat.SAFETENSORS: 1.25,   # FP16 → slight VRAM overhead
        ModelFormat.PYTORCH_BIN: 1.30,
        ModelFormat.GGUF: 1.05,          # Already quantized, minimal overhead
        ModelFormat.GPTQ: 1.15,
        ModelFormat.AWQ: 1.10,
        ModelFormat.UNKNOWN: 1.50,
    }
    try:
        if model_path.is_file():
            disk_bytes = model_path.stat().st_size
        else:
            disk_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        return int(disk_bytes * multipliers.get(model_format, 1.5))
    except Exception:
        return 0


class ModelLoader:
    """Multi-format model lifecycle manager.

    Manages discovery, loading, caching (LRU by VRAM budget), warm-up,
    and unloading of self-hosted models. All state is in-process; persistence
    is handled by the upstream model registry service.

    The cache eviction policy is LRU ordered by last_used_at. When a new
    model would exceed max_vram_bytes, the least-recently-used model is
    unloaded to reclaim memory before loading proceeds.
    """

    def __init__(
        self,
        model_root: str,
        max_vram_bytes: int = 24 * 1024 ** 3,  # 24 GiB default
        max_concurrent_loads: int = 2,
        warmup_prompt: str = "Hello, world!",
    ) -> None:
        """Initialize the model loader.

        Args:
            model_root: Root filesystem path where model directories live.
            max_vram_bytes: VRAM budget in bytes for the LRU cache.
            max_concurrent_loads: Maximum simultaneous model loads.
            warmup_prompt: Short prompt used for model warm-up prefill.
        """
        self._model_root = Path(model_root)
        self._max_vram_bytes = max_vram_bytes
        self._warmup_prompt = warmup_prompt
        self._semaphore = asyncio.Semaphore(max_concurrent_loads)

        # OrderedDict maintains LRU order: most-recent at the end
        self._cache: OrderedDict[str, ModelEntry] = OrderedDict()
        self._vram_used: int = 0
        self._loading_locks: dict[str, asyncio.Lock] = {}

        logger.info(
            "ModelLoader initialized",
            model_root=str(self._model_root),
            max_vram_gb=round(max_vram_bytes / 1024 ** 3, 1),
            max_concurrent_loads=max_concurrent_loads,
        )

    def discover_models(self) -> list[dict[str, Any]]:
        """Scan the model root directory for available models.

        Returns:
            List of dicts with model_id, format, path, and vram_estimate_bytes.
        """
        discovered: list[dict[str, Any]] = []
        if not self._model_root.exists():
            logger.warning("Model root does not exist", path=str(self._model_root))
            return discovered

        for entry in sorted(self._model_root.iterdir()):
            if entry.is_dir() or (entry.is_file() and entry.suffix.lower() in {".gguf", ".safetensors"}):
                model_id = entry.stem if entry.is_file() else entry.name
                fmt = _detect_format(entry)
                vram_estimate = _estimate_vram_bytes(entry, fmt)
                discovered.append(
                    {
                        "model_id": model_id,
                        "format": fmt.value,
                        "path": str(entry),
                        "vram_estimate_bytes": vram_estimate,
                    }
                )
                logger.debug(
                    "Discovered model",
                    model_id=model_id,
                    format=fmt.value,
                    vram_estimate_gb=round(vram_estimate / 1024 ** 3, 2),
                )

        return discovered

    async def load_model(self, model_id: str) -> ModelEntry:
        """Load a model into memory, evicting LRU entries if needed.

        Concurrent calls for the same model_id are serialized via a per-model
        lock, so the model is loaded exactly once.

        Args:
            model_id: Model identifier matching the directory name under model_root.

        Returns:
            ModelEntry describing the loaded model.

        Raises:
            FileNotFoundError: If the model path does not exist.
            RuntimeError: If loading fails after eviction attempts.
        """
        # Return already-loaded model, updating LRU position
        if model_id in self._cache and self._cache[model_id].status == ModelStatus.READY:
            self._cache.move_to_end(model_id)
            self._cache[model_id].last_used_at = time.monotonic()
            return self._cache[model_id]

        # Serialize per-model loading
        if model_id not in self._loading_locks:
            self._loading_locks[model_id] = asyncio.Lock()

        async with self._loading_locks[model_id]:
            # Double-check after acquiring lock
            if model_id in self._cache and self._cache[model_id].status == ModelStatus.READY:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]

            model_path = self._model_root / model_id
            if not model_path.exists():
                # Try as a direct file path (GGUF single-file models)
                gguf_path = self._model_root / f"{model_id}.gguf"
                if gguf_path.exists():
                    model_path = gguf_path
                else:
                    raise FileNotFoundError(f"Model not found: {model_id} (checked {model_path})")

            model_format = _detect_format(model_path)
            vram_needed = _estimate_vram_bytes(model_path, model_format)

            # Evict LRU entries until we have room
            await self._evict_until_fits(vram_needed)

            async with self._semaphore:
                entry = ModelEntry(
                    model_id=model_id,
                    model_format=model_format,
                    vram_bytes=vram_needed,
                    status=ModelStatus.LOADING,
                    loaded_at=time.monotonic(),
                    last_used_at=time.monotonic(),
                )
                self._cache[model_id] = entry

                try:
                    logger.info(
                        "Loading model",
                        model_id=model_id,
                        format=model_format.value,
                        vram_estimate_gb=round(vram_needed / 1024 ** 3, 2),
                    )
                    handle = await self._do_load(model_path, model_format)
                    entry.handle = handle
                    entry.status = ModelStatus.WARMING_UP
                    self._vram_used += vram_needed

                    # Run warm-up prefill
                    warmup_tokens = await self._warmup(entry)
                    entry.warmup_tokens_generated = warmup_tokens
                    entry.status = ModelStatus.READY
                    entry.loaded_at = time.monotonic()

                    logger.info(
                        "Model loaded and warmed up",
                        model_id=model_id,
                        warmup_tokens=warmup_tokens,
                        total_vram_gb=round(self._vram_used / 1024 ** 3, 2),
                    )

                except Exception as exc:
                    entry.status = ModelStatus.ERROR
                    entry.error_message = str(exc)
                    self._vram_used = max(0, self._vram_used - vram_needed)
                    logger.error(
                        "Model load failed",
                        model_id=model_id,
                        error=str(exc),
                    )
                    raise RuntimeError(f"Failed to load model {model_id}: {exc}") from exc

        return entry

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model and reclaim its VRAM.

        Args:
            model_id: Model to unload.

        Returns:
            True if the model was unloaded, False if it was not loaded.
        """
        if model_id not in self._cache:
            return False

        entry = self._cache[model_id]
        entry.status = ModelStatus.UNLOADING

        try:
            await self._do_unload(entry)
        except Exception as exc:
            logger.warning("Error during model unload", model_id=model_id, error=str(exc))
        finally:
            self._vram_used = max(0, self._vram_used - entry.vram_bytes)
            del self._cache[model_id]
            if model_id in self._loading_locks:
                del self._loading_locks[model_id]

        logger.info(
            "Model unloaded",
            model_id=model_id,
            freed_vram_gb=round(entry.vram_bytes / 1024 ** 3, 2),
            remaining_vram_gb=round(self._vram_used / 1024 ** 3, 2),
        )
        return True

    def get_model_status(self, model_id: str) -> ModelEntry | None:
        """Get the current status of a loaded model.

        Args:
            model_id: Model to query.

        Returns:
            ModelEntry if loaded, None if not in cache.
        """
        return self._cache.get(model_id)

    def list_loaded_models(self) -> list[dict[str, Any]]:
        """List all currently loaded models with their status.

        Returns:
            List of dicts describing each loaded model.
        """
        return [
            {
                "model_id": entry.model_id,
                "format": entry.model_format.value,
                "status": entry.status.value,
                "vram_bytes": entry.vram_bytes,
                "vram_gb": round(entry.vram_bytes / 1024 ** 3, 2),
                "loaded_at": entry.loaded_at,
                "last_used_at": entry.last_used_at,
                "warmup_tokens_generated": entry.warmup_tokens_generated,
                "error_message": entry.error_message,
            }
            for entry in self._cache.values()
        ]

    def get_vram_usage(self) -> dict[str, int | float]:
        """Return current VRAM utilization.

        Returns:
            Dict with used_bytes, total_bytes, and utilization_pct.
        """
        return {
            "used_bytes": self._vram_used,
            "total_bytes": self._max_vram_bytes,
            "utilization_pct": round(self._vram_used / max(self._max_vram_bytes, 1) * 100, 2),
        }

    async def _evict_until_fits(self, required_bytes: int) -> None:
        """Evict LRU models until there is room for required_bytes.

        Args:
            required_bytes: VRAM bytes needed for the next model load.
        """
        while self._vram_used + required_bytes > self._max_vram_bytes and self._cache:
            # Pop LRU (first item in OrderedDict)
            lru_id, lru_entry = next(iter(self._cache.items()))
            if lru_entry.status != ModelStatus.READY:
                # Skip models that are not in a stable state
                break
            logger.info(
                "Evicting LRU model to free VRAM",
                evicted_model=lru_id,
                freed_vram_gb=round(lru_entry.vram_bytes / 1024 ** 3, 2),
            )
            await self.unload_model(lru_id)

    async def _do_load(self, model_path: Path, model_format: ModelFormat) -> Any:
        """Perform the actual model loading (format-dispatched).

        This method runs in an executor thread to avoid blocking the event loop
        during heavy I/O and model initialization.

        Args:
            model_path: Filesystem path to the model.
            model_format: Detected serialization format.

        Returns:
            Opaque model handle (framework-specific).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_sync,
            model_path,
            model_format,
        )

    def _load_sync(self, model_path: Path, model_format: ModelFormat) -> dict[str, Any]:
        """Synchronous model load dispatched per format.

        Args:
            model_path: Filesystem path to the model.
            model_format: Detected serialization format.

        Returns:
            Dict acting as a lightweight handle with load metadata.
        """
        # Production implementations would call:
        #   SAFETENSORS/PYTORCH_BIN → transformers.AutoModelForCausalLM.from_pretrained
        #   GGUF → llama_cpp.Llama
        #   GPTQ → auto_gptq.AutoGPTQForCausalLM.from_quantized
        #   AWQ → awq.AutoAWQForCausalLM.from_quantized
        #
        # For framework independence, we return a metadata handle.
        # Callers that need actual inference should use VLLMProvider or OllamaProvider.
        return {
            "path": str(model_path),
            "format": model_format.value,
            "load_time": time.monotonic(),
            "model_id": model_path.stem,
        }

    async def _warmup(self, entry: ModelEntry) -> int:
        """Run a warm-up prefill pass to initialize KV-cache.

        Args:
            entry: ModelEntry for the model to warm up.

        Returns:
            Number of tokens generated during warm-up.
        """
        # In production, this would call entry.handle to run one forward pass
        # with self._warmup_prompt and count generated tokens.
        await asyncio.sleep(0)  # Yield to event loop
        return len(self._warmup_prompt.split())

    async def _do_unload(self, entry: ModelEntry) -> None:
        """Release model resources from memory.

        Args:
            entry: ModelEntry to unload.
        """
        # In production, this calls framework-specific cleanup:
        #   del model; torch.cuda.empty_cache()
        entry.handle = None
        await asyncio.sleep(0)
