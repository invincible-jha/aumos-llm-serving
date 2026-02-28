"""LLM serving benchmark script — measures latency, throughput, and TTFT.

Gap #136: Benchmark data for competitive positioning.

Usage:
    python scripts/benchmark_latency.py --base-url http://localhost:8000 \
        --model claude-sonnet-4-6 --concurrency 10 --requests 100

Metrics captured:
- Time-to-first-token (TTFT) for streaming requests
- Total latency (p50, p90, p95, p99)
- Throughput (tokens/second, requests/second)
- Cost per 1M tokens (from usage metadata)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class BenchmarkResult:
    """Metrics for a single benchmark request."""

    request_id: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""

    model: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_ttft_ms: float
    p95_ttft_ms: float
    avg_tokens_per_second: float
    total_tokens: int
    requests_per_second: float
    wall_time_seconds: float
    errors: list[str] = field(default_factory=list)


_BENCHMARK_PROMPTS = [
    "Summarize the key principles of machine learning in 3 sentences.",
    "What are the main differences between supervised and unsupervised learning?",
    "Explain the concept of gradient descent in simple terms.",
    "List 5 common use cases for large language models in enterprise settings.",
    "What is the difference between precision and recall in machine learning?",
]


async def run_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    request_id: int,
    stream: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark request and collect metrics."""
    result = BenchmarkResult(request_id=request_id, model=model)
    start_time = time.perf_counter()

    try:
        if stream:
            ttft_captured = False
            tokens_received = 0

            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "max_tokens": 200,
                },
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk: dict[str, Any] = json.loads(data_str)
                            if not ttft_captured:
                                result.ttft_ms = (time.perf_counter() - start_time) * 1000
                                ttft_captured = True
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                tokens_received += 1
                        except json.JSONDecodeError:
                            pass
            result.completion_tokens = tokens_received
        else:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        if result.total_latency_ms > 0:
            result.tokens_per_second = result.completion_tokens / (result.total_latency_ms / 1000)

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        result.total_latency_ms = (time.perf_counter() - start_time) * 1000

    return result


async def run_benchmark(
    base_url: str,
    model: str,
    concurrency: int,
    total_requests: int,
    stream: bool = True,
) -> BenchmarkSummary:
    """Run the full benchmark with concurrent requests."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[BenchmarkResult] = []
    wall_start = time.perf_counter()

    async def bounded_request(request_id: int) -> BenchmarkResult:
        async with semaphore:
            prompt = _BENCHMARK_PROMPTS[request_id % len(_BENCHMARK_PROMPTS)]
            async with httpx.AsyncClient() as client:
                return await run_single_request(
                    client, base_url, model, prompt, request_id, stream=stream
                )

    tasks = [bounded_request(i) for i in range(total_requests)]
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - wall_start

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    latencies = [r.total_latency_ms for r in successful]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return BenchmarkSummary(
        model=model,
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_ms=percentile(sorted_latencies, 50),
        p90_latency_ms=percentile(sorted_latencies, 90),
        p95_latency_ms=percentile(sorted_latencies, 95),
        p99_latency_ms=percentile(sorted_latencies, 99),
        avg_ttft_ms=statistics.mean(ttfts) if ttfts else 0.0,
        p95_ttft_ms=percentile(sorted(ttfts), 95),
        avg_tokens_per_second=statistics.mean([r.tokens_per_second for r in successful]) if successful else 0.0,
        total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in successful),
        requests_per_second=len(successful) / wall_time if wall_time > 0 else 0.0,
        wall_time_seconds=wall_time,
        errors=[r.error for r in failed if r.error],
    )


def print_summary(summary: BenchmarkSummary) -> None:
    """Print a formatted benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — {summary.model}")
    print(f"{'='*60}")
    print(f"Total requests:      {summary.total_requests}")
    print(f"Successful:          {summary.successful_requests}")
    print(f"Failed:              {summary.failed_requests}")
    print(f"Wall time:           {summary.wall_time_seconds:.2f}s")
    print(f"Requests/sec:        {summary.requests_per_second:.2f}")
    print(f"\nLatency (ms):")
    print(f"  avg:               {summary.avg_latency_ms:.1f}")
    print(f"  p50:               {summary.p50_latency_ms:.1f}")
    print(f"  p90:               {summary.p90_latency_ms:.1f}")
    print(f"  p95:               {summary.p95_latency_ms:.1f}")
    print(f"  p99:               {summary.p99_latency_ms:.1f}")
    print(f"\nTime-to-first-token (ms):")
    print(f"  avg:               {summary.avg_ttft_ms:.1f}")
    print(f"  p95:               {summary.p95_ttft_ms:.1f}")
    print(f"\nThroughput:")
    print(f"  avg tokens/sec:    {summary.avg_tokens_per_second:.1f}")
    print(f"  total tokens:      {summary.total_tokens}")
    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for error in summary.errors[:5]:
            print(f"  - {error}")
    print(f"{'='*60}\n")


async def main() -> None:
    """Entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="AumOS LLM Serving benchmark tool")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the LLM serving service")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to benchmark")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming completions")
    parser.add_argument("--output-json", help="Write results to JSON file")
    args = parser.parse_args()

    print(f"Starting benchmark: model={args.model}, concurrency={args.concurrency}, requests={args.requests}")

    summary = await run_benchmark(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        total_requests=args.requests,
        stream=not args.no_stream,
    )

    print_summary(summary)

    if args.output_json:
        import dataclasses

        with open(args.output_json, "w") as f:
            json.dump(dataclasses.asdict(summary), f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
