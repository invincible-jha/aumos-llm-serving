"""Complexity-based LLM routing with cost and latency budget awareness.

Classifies prompt complexity as a float in [0.0, 1.0] using pure-Python
feature extraction (no external ML model) to keep classification latency
under 5 ms. Routes the request to the most cost-effective model capable
of handling the detected complexity level.

Complexity tiers:
  0.0–0.3   simple      → local / cheapest model
  0.3–0.7   moderate    → mid-tier model
  0.7–1.0   complex     → premium model

Usage:
    router = ComplexityRouter(
        local_model="ollama/llama3.2",
        mid_tier_model="gpt-4o-mini",
        premium_model="claude-opus-4",
    )
    decision = router.route(
        prompt="Explain quantum entanglement at a PhD level...",
        tenant_id="tenant-abc",
    )
    print(decision.selected_model)
"""
from __future__ import annotations

import re
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelTarget(BaseModel):
    """A candidate model that the complexity router may route to.

    Attributes:
        model_id: Provider-qualified model identifier (e.g., 'ollama/llama3.2').
        tier: Complexity tier this model serves: 'local', 'mid', or 'premium'.
        max_context_tokens: Maximum context window in tokens.
        cost_per_million_input_tokens: Input token cost in USD per 1 M tokens.
        cost_per_million_output_tokens: Output token cost in USD per 1 M tokens.
        typical_latency_ms: Typical time-to-first-token in milliseconds.
    """

    model_id: str = Field(description="Provider-qualified model identifier")
    tier: str = Field(description="Complexity tier: local | mid | premium")
    max_context_tokens: int = Field(default=8192, ge=512)
    cost_per_million_input_tokens: float = Field(default=0.0, ge=0.0)
    cost_per_million_output_tokens: float = Field(default=0.0, ge=0.0)
    typical_latency_ms: int = Field(default=500, ge=0)


class ModelPreferences(BaseModel):
    """Tenant-specific model preferences that override the default routing table.

    Attributes:
        local_model: Model ID to use for simple prompts (complexity 0.0–0.3).
        mid_tier_model: Model ID for moderate prompts (complexity 0.3–0.7).
        premium_model: Model ID for complex prompts (complexity 0.7–1.0).
        latency_budget_ms: Maximum acceptable latency. Requests exceeding this
            may be downgraded to a faster model tier.
        cost_ceiling_usd: Maximum cost per request. Requests projected above
            this threshold are downgraded to a cheaper tier.
    """

    local_model: str | None = Field(default=None)
    mid_tier_model: str | None = Field(default=None)
    premium_model: str | None = Field(default=None)
    latency_budget_ms: int | None = Field(default=None, ge=0)
    cost_ceiling_usd: float | None = Field(default=None, ge=0.0)


class RoutingDecision(BaseModel):
    """The result of a complexity routing evaluation.

    Attributes:
        selected_model: The model ID chosen for this request.
        tier: The complexity tier that was selected.
        complexity_score: Raw complexity score in [0.0, 1.0].
        latency_budget_ms: Effective latency budget that was applied.
        cost_ceiling_usd: Effective cost ceiling that was applied.
        downgraded: True if the router downgraded from the natural tier due to
            latency or cost constraints.
        reasoning: Human-readable explanation of the routing decision.
        features: Raw feature values extracted from the prompt (for debugging).
    """

    selected_model: str
    tier: str
    complexity_score: float = Field(ge=0.0, le=1.0)
    latency_budget_ms: int | None = None
    cost_ceiling_usd: float | None = None
    downgraded: bool = False
    reasoning: str = ""
    features: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction helpers — all run in pure Python with no I/O
# ---------------------------------------------------------------------------


_CODE_BLOCK_PATTERN: re.Pattern[str] = re.compile(r"```[\s\S]*?```|`[^`]+`")
_REASONING_MARKERS: frozenset[str] = frozenset({
    "step by step",
    "step-by-step",
    "explain why",
    "reason through",
    "chain of thought",
    "think through",
    "analyze",
    "critique",
    "compare and contrast",
    "pros and cons",
    "trade-offs",
    "tradeoffs",
    "multi-step",
    "break down",
})
_QUESTION_MARKERS: frozenset[str] = frozenset({
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "explain",
    "describe",
    "define",
    "list",
    "summarize",
    "translate",
    "calculate",
    "compute",
})
_COMPLEX_DOMAIN_TERMS: frozenset[str] = frozenset({
    "quantum",
    "cryptography",
    "differential equation",
    "topology",
    "bayesian",
    "adversarial",
    "reinforcement learning",
    "transformer architecture",
    "attention mechanism",
    "gradient descent",
    "backpropagation",
    "eigenvector",
    "stochastic",
    "markov",
    "recursive",
    "algorithm complexity",
    "big-o",
    "formal proof",
    "theorem",
    "lemma",
    "hypothesis",
    "regression",
    "regression analysis",
})


def _extract_features(prompt: str) -> dict[str, float]:
    """Extract complexity features from a prompt string.

    All operations are O(n) string scans designed to run well under 5 ms even
    for prompts of 8 000+ characters.

    Args:
        prompt: The full user prompt text.

    Returns:
        Dictionary mapping feature name to its normalised value in [0.0, 1.0].
    """
    prompt_lower = prompt.lower()
    words = prompt_lower.split()
    word_count = max(len(words), 1)

    # 1. Length score — longer prompts tend to be more complex
    #    Cap at 2 000 words for normalization
    length_score: float = min(word_count / 2_000, 1.0)

    # 2. Vocabulary diversity — unique tokens / total tokens
    unique_words = set(words)
    vocabulary_diversity: float = len(unique_words) / word_count

    # 3. Code presence — any code block bumps complexity up
    code_matches = _CODE_BLOCK_PATTERN.findall(prompt)
    code_score: float = min(len(code_matches) * 0.3, 1.0)

    # 4. Multi-step reasoning markers
    reasoning_count = sum(1 for marker in _REASONING_MARKERS if marker in prompt_lower)
    reasoning_score: float = min(reasoning_count * 0.25, 1.0)

    # 5. Question depth — number of distinct question words
    question_count = sum(1 for marker in _QUESTION_MARKERS if marker in prompt_lower)
    question_score: float = min(question_count / len(_QUESTION_MARKERS), 1.0)

    # 6. Complex domain vocabulary
    domain_count = sum(1 for term in _COMPLEX_DOMAIN_TERMS if term in prompt_lower)
    domain_score: float = min(domain_count * 0.2, 1.0)

    return {
        "length_score": length_score,
        "vocabulary_diversity": vocabulary_diversity,
        "code_score": code_score,
        "reasoning_score": reasoning_score,
        "question_score": question_score,
        "domain_score": domain_score,
    }


def _compute_complexity(features: dict[str, float]) -> float:
    """Compute an aggregate complexity score from extracted features.

    Weights are tuned to reflect empirical correlation of each feature with
    actual task difficulty (no external data source required at runtime).

    Args:
        features: Output from _extract_features().

    Returns:
        Complexity score in [0.0, 1.0].
    """
    weights: dict[str, float] = {
        "length_score": 0.15,
        "vocabulary_diversity": 0.20,
        "code_score": 0.20,
        "reasoning_score": 0.25,
        "question_score": 0.10,
        "domain_score": 0.10,
    }
    score = sum(features[key] * weight for key, weight in weights.items())
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# ComplexityRouter
# ---------------------------------------------------------------------------

# Tier boundary constants — exported for testing
SIMPLE_THRESHOLD: float = 0.30
COMPLEX_THRESHOLD: float = 0.70


class ComplexityRouter:
    """Route LLM requests to the appropriate model tier based on prompt complexity.

    Classification runs entirely in Python (no ML model) and targets < 5 ms
    total latency. The router also respects per-request latency and cost budgets,
    downgrading to a cheaper/faster model when constraints cannot be met by the
    naturally selected tier.

    Args:
        local_model: Model ID for simple prompts (complexity 0.0–0.3).
        mid_tier_model: Model ID for moderate prompts (complexity 0.3–0.7).
        premium_model: Model ID for complex prompts (complexity 0.7–1.0).
        model_catalog: Optional dict mapping model IDs to ModelTarget metadata.
            Used for latency/cost constraint evaluation. If omitted, constraints
            are not enforced and the natural complexity tier is always used.
    """

    # Default model tiers — overridden per-tenant via ModelPreferences
    DEFAULT_LOCAL_MODEL: str = "ollama/llama3.2"
    DEFAULT_MID_TIER_MODEL: str = "gpt-4o-mini"
    DEFAULT_PREMIUM_MODEL: str = "claude-opus-4"

    def __init__(
        self,
        local_model: str = DEFAULT_LOCAL_MODEL,
        mid_tier_model: str = DEFAULT_MID_TIER_MODEL,
        premium_model: str = DEFAULT_PREMIUM_MODEL,
        model_catalog: dict[str, ModelTarget] | None = None,
    ) -> None:
        """Initialise the ComplexityRouter.

        Args:
            local_model: Default model ID for simple prompts.
            mid_tier_model: Default model ID for moderate prompts.
            premium_model: Default model ID for complex prompts.
            model_catalog: Optional mapping of model_id → ModelTarget for
                latency/cost constraint checks.
        """
        self._local_model = local_model
        self._mid_tier_model = mid_tier_model
        self._premium_model = premium_model
        self._model_catalog: dict[str, ModelTarget] = model_catalog or {}

    def _tier_for_score(self, score: float) -> tuple[str, str]:
        """Map a complexity score to a (tier_name, model_id) pair.

        Args:
            score: Complexity score in [0.0, 1.0].

        Returns:
            Tuple of (tier_name, default_model_id).
        """
        if score < SIMPLE_THRESHOLD:
            return "local", self._local_model
        if score < COMPLEX_THRESHOLD:
            return "mid", self._mid_tier_model
        return "premium", self._premium_model

    def _apply_preferences(
        self,
        tier: str,
        model_id: str,
        preferences: ModelPreferences | None,
    ) -> tuple[str, str]:
        """Override the default model with tenant-specific preferences if set.

        Args:
            tier: The naturally selected tier.
            model_id: The default model ID for that tier.
            preferences: Optional per-tenant model preferences.

        Returns:
            Updated (tier, model_id) pair.
        """
        if preferences is None:
            return tier, model_id
        overrides: dict[str, str | None] = {
            "local": preferences.local_model,
            "mid": preferences.mid_tier_model,
            "premium": preferences.premium_model,
        }
        override = overrides.get(tier)
        return tier, override if override else model_id

    def _check_constraints(
        self,
        tier: str,
        model_id: str,
        preferences: ModelPreferences | None,
    ) -> tuple[str, str, bool]:
        """Downgrade model tier when latency or cost constraints cannot be met.

        Only applies when model metadata is available in the catalog.

        Args:
            tier: The currently selected tier.
            model_id: The currently selected model ID.
            preferences: Constraints to enforce.

        Returns:
            Updated (tier, model_id, was_downgraded) triple.
        """
        if preferences is None or model_id not in self._model_catalog:
            return tier, model_id, False

        target = self._model_catalog[model_id]
        latency_ok = (
            preferences.latency_budget_ms is None
            or target.typical_latency_ms <= preferences.latency_budget_ms
        )
        cost_per_1k_input = target.cost_per_million_input_tokens / 1_000.0
        cost_ok = (
            preferences.cost_ceiling_usd is None
            or cost_per_1k_input <= preferences.cost_ceiling_usd
        )

        if latency_ok and cost_ok:
            return tier, model_id, False

        # Try to downgrade to next cheaper/faster tier
        downgrade_order: list[tuple[str, str]] = [
            ("mid", preferences.mid_tier_model or self._mid_tier_model),
            ("local", preferences.local_model or self._local_model),
        ]
        for candidate_tier, candidate_model in downgrade_order:
            if candidate_tier >= tier:
                continue  # Only consider cheaper tiers
            if candidate_model not in self._model_catalog:
                return candidate_tier, candidate_model, True
            candidate_target = self._model_catalog[candidate_model]
            if (
                preferences.latency_budget_ms is None
                or candidate_target.typical_latency_ms <= preferences.latency_budget_ms
            ):
                return candidate_tier, candidate_model, True

        # Cannot satisfy constraints — return current selection and log
        logger.warning(
            "complexity_router_constraints_unmet",
            tier=tier,
            model_id=model_id,
            latency_ok=latency_ok,
            cost_ok=cost_ok,
        )
        return tier, model_id, False

    def route(
        self,
        prompt: str,
        tenant_id: str = "",
        preferences: ModelPreferences | None = None,
        system_prompt: str = "",
    ) -> RoutingDecision:
        """Classify prompt complexity and select the optimal model.

        Args:
            prompt: The user prompt text to classify.
            tenant_id: Tenant identifier (for logging and future per-tenant stats).
            preferences: Optional per-tenant model overrides and constraints.
            system_prompt: Optional system prompt; included in complexity scoring.

        Returns:
            RoutingDecision with the selected model and explanation.
        """
        combined_text = f"{system_prompt} {prompt}".strip() if system_prompt else prompt
        features = _extract_features(combined_text)
        complexity_score = _compute_complexity(features)

        tier, model_id = self._tier_for_score(complexity_score)
        tier, model_id = self._apply_preferences(tier, model_id, preferences)
        tier, model_id, downgraded = self._check_constraints(tier, model_id, preferences)

        reasoning = (
            f"Complexity score {complexity_score:.3f} mapped to tier '{tier}'. "
            f"Selected model: {model_id}."
        )
        if downgraded:
            reasoning += " Downgraded from natural tier due to latency/cost constraints."

        logger.debug(
            "complexity_router_decision",
            tenant_id=tenant_id,
            complexity_score=complexity_score,
            tier=tier,
            selected_model=model_id,
            downgraded=downgraded,
        )

        return RoutingDecision(
            selected_model=model_id,
            tier=tier,
            complexity_score=complexity_score,
            latency_budget_ms=preferences.latency_budget_ms if preferences else None,
            cost_ceiling_usd=preferences.cost_ceiling_usd if preferences else None,
            downgraded=downgraded,
            reasoning=reasoning,
            features=features,
        )

    def classify(self, prompt: str, system_prompt: str = "") -> float:
        """Return only the complexity score, without performing routing.

        Useful for analytics and testing.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system prompt.

        Returns:
            Complexity score in [0.0, 1.0].
        """
        combined_text = f"{system_prompt} {prompt}".strip() if system_prompt else prompt
        features = _extract_features(combined_text)
        return _compute_complexity(features)
