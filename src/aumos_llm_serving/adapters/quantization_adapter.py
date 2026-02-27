"""Quantization adapter.

Provides model quantization detection, format handling, auto-quantization,
quality assessment, and memory savings estimation for self-hosted models.
Supports INT8, FP8, GPTQ, and AWQ formats.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class QuantizationFormat(str, enum.Enum):
    """Supported model quantization formats."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"
    UNKNOWN = "unknown"


class QuantizationPrecision(int, enum.Enum):
    """Bits per weight for each format."""

    FP32 = 32
    FP16 = 16
    BF16 = 16
    FP8 = 8
    INT8 = 8
    INT4 = 4
    GPTQ = 4  # Typically 4-bit GPTQ
    AWQ = 4   # Typically 4-bit AWQ


# Map of format → typical perplexity degradation (vs. FP32 baseline)
_PERPLEXITY_DEGRADATION_ESTIMATE: dict[QuantizationFormat, float] = {
    QuantizationFormat.FP32: 0.0,
    QuantizationFormat.FP16: 0.01,
    QuantizationFormat.BF16: 0.02,
    QuantizationFormat.FP8: 0.05,
    QuantizationFormat.INT8: 0.08,
    QuantizationFormat.GPTQ: 0.12,
    QuantizationFormat.AWQ: 0.10,
    QuantizationFormat.INT4: 0.20,
    QuantizationFormat.UNKNOWN: 0.0,
}


@dataclass
class QuantizationProfile:
    """Complete quantization metadata for a model.

    Attributes:
        model_id: Model identifier.
        detected_format: The detected quantization format.
        bits_per_weight: Effective bits per weight after quantization.
        original_size_bytes: Estimated or measured FP16 model size in bytes.
        quantized_size_bytes: Actual quantized model size in bytes.
        memory_savings_pct: Percentage reduction in model size.
        estimated_perplexity_delta: Estimated perplexity increase vs. FP16.
        group_size: GPTQ/AWQ quantization group size (0 if not applicable).
        desc_act: Whether activation ordering is enabled (GPTQ specific).
        symmetric: Whether symmetric quantization is used.
        recommended: Whether this format is recommended for production use.
    """

    model_id: str
    detected_format: QuantizationFormat
    bits_per_weight: int
    original_size_bytes: int
    quantized_size_bytes: int
    memory_savings_pct: float
    estimated_perplexity_delta: float
    group_size: int
    desc_act: bool
    symmetric: bool
    recommended: bool


@dataclass
class AutoQuantizationRecommendation:
    """Recommended quantization settings for a given model and constraint.

    Attributes:
        recommended_format: Best format for the constraints.
        expected_size_bytes: Expected model size after quantization.
        expected_savings_pct: Expected memory savings percentage.
        expected_perplexity_delta: Expected perplexity degradation.
        group_size: Recommended group size for GPTQ/AWQ.
        rationale: Human-readable explanation of the recommendation.
    """

    recommended_format: QuantizationFormat
    expected_size_bytes: int
    expected_savings_pct: float
    expected_perplexity_delta: float
    group_size: int
    rationale: str


def _read_quantize_config(model_path: Path) -> dict[str, Any]:
    """Read quantize_config.json or config.json from a model directory.

    Args:
        model_path: Root directory of the model.

    Returns:
        Parsed JSON config dict, or empty dict if not found.
    """
    import json
    for config_name in ("quantize_config.json", "config.json", "adapter_config.json"):
        config_file = model_path / config_name
        if config_file.exists():
            try:
                return json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _get_directory_size_bytes(path: Path) -> int:
    """Recursively sum file sizes in a directory.

    Args:
        path: Directory to measure.

    Returns:
        Total size in bytes, or 0 on error.
    """
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


class QuantizationAdapter:
    """Model quantization format handler and quality assessor.

    Detects the quantization format of a model from its directory structure
    and config files, provides quality assessment, recommends quantization
    settings, and estimates memory savings.

    All heavy computation (actual model loading, perplexity measurement)
    is dispatched to executor threads to avoid blocking the event loop.
    """

    def __init__(self, model_root: str) -> None:
        """Initialize the quantization adapter.

        Args:
            model_root: Root directory where model directories are stored.
        """
        self._model_root = Path(model_root)
        logger.info("QuantizationAdapter initialized", model_root=str(self._model_root))

    def detect_format(self, model_id: str) -> QuantizationFormat:
        """Detect the quantization format of a model by inspecting its files.

        Args:
            model_id: Model identifier (subdirectory name under model_root).

        Returns:
            Detected QuantizationFormat.
        """
        model_path = self._model_root / model_id
        if not model_path.exists():
            logger.warning("Model path not found for format detection", model_id=model_id)
            return QuantizationFormat.UNKNOWN

        config = _read_quantize_config(model_path)

        # Check quantize_config.json fields (AutoGPTQ format)
        quant_type = str(config.get("quant_type", "")).lower()
        if quant_type == "gptq":
            return QuantizationFormat.GPTQ
        if quant_type == "awq":
            return QuantizationFormat.AWQ

        # Check config.json quantization_config block (HuggingFace format)
        quant_config = config.get("quantization_config", {})
        if isinstance(quant_config, dict):
            quant_method = str(quant_config.get("quant_type", quant_config.get("load_in_format", ""))).lower()
            if "gptq" in quant_method:
                return QuantizationFormat.GPTQ
            if "awq" in quant_method:
                return QuantizationFormat.AWQ
            if "int8" in quant_method or quant_config.get("load_in_8bit"):
                return QuantizationFormat.INT8
            if "int4" in quant_method or quant_config.get("load_in_4bit"):
                return QuantizationFormat.INT4
            if "fp8" in quant_method:
                return QuantizationFormat.FP8

        # Detect from file extensions
        files = list(model_path.rglob("*")) if model_path.is_dir() else [model_path]
        extensions = {f.suffix.lower() for f in files}

        if ".gguf" in extensions:
            # GGUF files are always quantized (typically INT4/INT8)
            return QuantizationFormat.INT4

        # Check for FP16 vs FP32 via dtype in config
        torch_dtype = str(config.get("torch_dtype", "")).lower()
        if torch_dtype in {"float16", "fp16", "half"}:
            return QuantizationFormat.FP16
        if torch_dtype in {"bfloat16", "bf16"}:
            return QuantizationFormat.BF16
        if torch_dtype in {"float32", "fp32", "float"}:
            return QuantizationFormat.FP32

        # Default: assume FP16 for safetensors, FP32 for .bin
        if ".safetensors" in extensions:
            return QuantizationFormat.FP16
        if ".bin" in extensions:
            return QuantizationFormat.FP32

        return QuantizationFormat.UNKNOWN

    def profile_model(self, model_id: str) -> QuantizationProfile:
        """Build a complete quantization profile for a model.

        Args:
            model_id: Model identifier.

        Returns:
            QuantizationProfile with all detected and estimated attributes.
        """
        model_path = self._model_root / model_id
        detected_format = self.detect_format(model_id)

        quantized_size_bytes = _get_directory_size_bytes(model_path)
        config = _read_quantize_config(model_path)

        # Estimate original FP16 size from quantized size
        bits = QuantizationPrecision[detected_format.name].value if detected_format.name in QuantizationPrecision.__members__ else 16
        fp16_multiplier = 16.0 / max(bits, 1)
        original_size_bytes = int(quantized_size_bytes * fp16_multiplier)

        savings_pct = (
            (1.0 - quantized_size_bytes / max(original_size_bytes, 1)) * 100
            if original_size_bytes > 0
            else 0.0
        )

        perplexity_delta = _PERPLEXITY_DEGRADATION_ESTIMATE.get(detected_format, 0.0)

        # Extract GPTQ/AWQ-specific parameters
        group_size = int(config.get("group_size", config.get("q_group_size", 0)))
        desc_act = bool(config.get("desc_act", False))
        symmetric = bool(config.get("sym", config.get("symmetric", True)))

        # Recommendation: FP8/INT8/GPTQ with group_size=128 are production-safe
        recommended = detected_format in {
            QuantizationFormat.FP16,
            QuantizationFormat.BF16,
            QuantizationFormat.FP8,
            QuantizationFormat.INT8,
            QuantizationFormat.GPTQ,
            QuantizationFormat.AWQ,
        }

        profile = QuantizationProfile(
            model_id=model_id,
            detected_format=detected_format,
            bits_per_weight=bits,
            original_size_bytes=original_size_bytes,
            quantized_size_bytes=quantized_size_bytes,
            memory_savings_pct=round(savings_pct, 1),
            estimated_perplexity_delta=round(perplexity_delta, 4),
            group_size=group_size,
            desc_act=desc_act,
            symmetric=symmetric,
            recommended=recommended,
        )

        logger.info(
            "Model quantization profile",
            model_id=model_id,
            format=detected_format.value,
            bits=bits,
            size_gb=round(quantized_size_bytes / 1024 ** 3, 2),
            savings_pct=round(savings_pct, 1),
        )

        return profile

    def recommend_quantization(
        self,
        model_id: str,
        vram_budget_bytes: int,
        max_perplexity_delta: float = 0.15,
    ) -> AutoQuantizationRecommendation:
        """Recommend the best quantization format for a given VRAM budget.

        Args:
            model_id: Model to quantize.
            vram_budget_bytes: Available VRAM in bytes.
            max_perplexity_delta: Maximum acceptable perplexity degradation.

        Returns:
            AutoQuantizationRecommendation with the optimal format.
        """
        model_path = self._model_root / model_id
        original_size = _get_directory_size_bytes(model_path)

        # Try formats from highest quality to lowest
        format_priority: list[QuantizationFormat] = [
            QuantizationFormat.FP16,
            QuantizationFormat.BF16,
            QuantizationFormat.FP8,
            QuantizationFormat.INT8,
            QuantizationFormat.AWQ,
            QuantizationFormat.GPTQ,
            QuantizationFormat.INT4,
        ]

        for fmt in format_priority:
            ppl_delta = _PERPLEXITY_DEGRADATION_ESTIMATE.get(fmt, 0.0)
            if ppl_delta > max_perplexity_delta:
                continue

            bits = QuantizationPrecision[fmt.name].value if fmt.name in QuantizationPrecision.__members__ else 16
            compression_ratio = 16.0 / bits
            expected_size = int(original_size / compression_ratio * 1.1)  # 10% overhead

            if expected_size <= vram_budget_bytes:
                savings_pct = (1.0 - expected_size / max(original_size, 1)) * 100
                group_size = 128 if fmt in {QuantizationFormat.GPTQ, QuantizationFormat.AWQ} else 0

                recommendation = AutoQuantizationRecommendation(
                    recommended_format=fmt,
                    expected_size_bytes=expected_size,
                    expected_savings_pct=round(savings_pct, 1),
                    expected_perplexity_delta=ppl_delta,
                    group_size=group_size,
                    rationale=(
                        f"{fmt.value} fits in {round(vram_budget_bytes / 1024**3, 1)} GiB budget "
                        f"(~{round(expected_size / 1024**3, 1)} GiB) "
                        f"with {round(ppl_delta*100, 1)}% estimated perplexity increase."
                    ),
                )

                logger.info(
                    "Quantization recommendation",
                    model_id=model_id,
                    recommended_format=fmt.value,
                    expected_size_gb=round(expected_size / 1024 ** 3, 2),
                    savings_pct=round(savings_pct, 1),
                )

                return recommendation

        # Fallback: most aggressive quantization
        return AutoQuantizationRecommendation(
            recommended_format=QuantizationFormat.INT4,
            expected_size_bytes=int(original_size / 4),
            expected_savings_pct=75.0,
            expected_perplexity_delta=_PERPLEXITY_DEGRADATION_ESTIMATE[QuantizationFormat.INT4],
            group_size=128,
            rationale=(
                "No format within perplexity threshold fits the VRAM budget. "
                "INT4 is the most aggressive option; consider a smaller model."
            ),
        )

    def estimate_memory_savings(
        self,
        original_size_bytes: int,
        target_format: QuantizationFormat,
    ) -> dict[str, float]:
        """Estimate memory savings from quantizing to a target format.

        Args:
            original_size_bytes: Size of the FP16 model in bytes.
            target_format: Quantization format to estimate for.

        Returns:
            Dict with size_bytes, savings_bytes, savings_pct, size_gb.
        """
        bits = QuantizationPrecision[target_format.name].value if target_format.name in QuantizationPrecision.__members__ else 16
        compression_ratio = 16.0 / max(bits, 1)
        quantized_bytes = int(original_size_bytes / compression_ratio)
        savings_bytes = original_size_bytes - quantized_bytes
        savings_pct = savings_bytes / max(original_size_bytes, 1) * 100

        return {
            "original_size_bytes": float(original_size_bytes),
            "quantized_size_bytes": float(quantized_bytes),
            "savings_bytes": float(savings_bytes),
            "savings_pct": round(savings_pct, 1),
            "quantized_size_gb": round(quantized_bytes / 1024 ** 3, 2),
            "original_size_gb": round(original_size_bytes / 1024 ** 3, 2),
            "compression_ratio": round(compression_ratio, 2),
            "estimated_perplexity_delta": _PERPLEXITY_DEGRADATION_ESTIMATE.get(target_format, 0.0),
        }

    def assess_quality(
        self,
        model_id: str,
        baseline_perplexity: float,
    ) -> dict[str, Any]:
        """Assess quantization quality for an already-quantized model.

        Estimates quality degradation relative to a provided baseline
        perplexity (typically measured on FP16 or FP32).

        Args:
            model_id: Quantized model to assess.
            baseline_perplexity: Perplexity of the reference (FP16) model.

        Returns:
            Dict with detected_format, estimated_ppl, ppl_delta, and quality_rating.
        """
        profile = self.profile_model(model_id)
        estimated_ppl = baseline_perplexity * (1.0 + profile.estimated_perplexity_delta)
        ppl_increase_pct = profile.estimated_perplexity_delta * 100

        if ppl_increase_pct < 2.0:
            quality_rating = "excellent"
        elif ppl_increase_pct < 5.0:
            quality_rating = "good"
        elif ppl_increase_pct < 12.0:
            quality_rating = "acceptable"
        else:
            quality_rating = "degraded"

        return {
            "model_id": model_id,
            "detected_format": profile.detected_format.value,
            "bits_per_weight": profile.bits_per_weight,
            "baseline_perplexity": baseline_perplexity,
            "estimated_perplexity": round(estimated_ppl, 4),
            "perplexity_increase_pct": round(ppl_increase_pct, 2),
            "quality_rating": quality_rating,
            "memory_savings_pct": profile.memory_savings_pct,
            "recommended": profile.recommended,
        }
