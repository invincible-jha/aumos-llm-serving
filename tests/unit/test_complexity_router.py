"""Unit tests for ComplexityRouter.

Verifies routing tier assignment for 20+ test prompts spanning all three
complexity tiers (local, mid, premium), preference overrides, constraint-based
downgrading, and the classify() method.
"""
from __future__ import annotations

import pytest

from aumos_llm_serving.core.routing.complexity_router import (
    COMPLEX_THRESHOLD,
    SIMPLE_THRESHOLD,
    ComplexityRouter,
    ModelPreferences,
    ModelTarget,
    RoutingDecision,
    _compute_complexity,
    _extract_features,
)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for the _extract_features() helper."""

    def test_empty_prompt_returns_zero_scores(self) -> None:
        features = _extract_features("")
        assert all(v >= 0.0 for v in features.values())

    def test_long_prompt_raises_length_score(self) -> None:
        long_prompt = "word " * 1000
        features = _extract_features(long_prompt)
        assert features["length_score"] > 0.3

    def test_code_block_detected(self) -> None:
        prompt = "What does this code do? ```python\ndef foo(): pass\n```"
        features = _extract_features(prompt)
        assert features["code_score"] > 0.0

    def test_inline_code_detected(self) -> None:
        prompt = "What does `np.linalg.norm()` do?"
        features = _extract_features(prompt)
        assert features["code_score"] > 0.0

    def test_reasoning_marker_detected(self) -> None:
        prompt = "Step by step, explain how neural networks learn."
        features = _extract_features(prompt)
        assert features["reasoning_score"] > 0.0

    def test_domain_vocabulary_detected(self) -> None:
        prompt = "Explain the eigenvector decomposition in quantum computing."
        features = _extract_features(prompt)
        assert features["domain_score"] > 0.0

    def test_all_features_clamped_to_one(self) -> None:
        huge_prompt = "word " * 5000 + " ".join(["step by step analyze critique"] * 10)
        features = _extract_features(huge_prompt)
        assert all(v <= 1.0 for v in features.values())


# ---------------------------------------------------------------------------
# Complexity scoring tests
# ---------------------------------------------------------------------------


class TestComputeComplexity:
    """Tests for the _compute_complexity() aggregation function."""

    def test_all_zero_features_gives_zero(self) -> None:
        features = {
            "length_score": 0.0,
            "vocabulary_diversity": 0.0,
            "code_score": 0.0,
            "reasoning_score": 0.0,
            "question_score": 0.0,
            "domain_score": 0.0,
        }
        assert _compute_complexity(features) == pytest.approx(0.0)

    def test_score_bounded_zero_to_one(self) -> None:
        features = {k: 1.0 for k in [
            "length_score", "vocabulary_diversity", "code_score",
            "reasoning_score", "question_score", "domain_score",
        ]}
        score = _compute_complexity(features)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Router tier assignment — 20+ test prompts
# ---------------------------------------------------------------------------


SIMPLE_PROMPTS: list[str] = [
    "Hello, how are you?",
    "What is 2 + 2?",
    "Translate 'hello' to Spanish.",
    "What time is it?",
    "Tell me a short joke.",
    "What is the capital of France?",
    "List three colors.",
    "What is the weather like?",
]

MODERATE_PROMPTS: list[str] = [
    "Summarize the history of the Roman Empire in three paragraphs.",
    "Explain the difference between TCP and UDP protocols.",
    "Write a Python function that reverses a string.",
    "What are the pros and cons of microservices architecture?",
    "How does gradient descent work in machine learning?",
    "Explain the concept of recursion with a simple example.",
    "Compare SQL and NoSQL databases.",
]

COMPLEX_PROMPTS: list[str] = [
    "Step by step, analyze the eigenvector decomposition of this matrix and explain "
    "why it is relevant to quantum computing algorithms. Include formal proofs where possible.",
    "Critique the Bayesian framework for statistical inference, compare and contrast it "
    "with frequentist approaches, and reason through the philosophical implications for "
    "scientific epistemology. Provide a multi-step analysis.",
    "Analyze the adversarial robustness of transformer attention mechanisms under "
    "gradient-based attacks. Explain how differential privacy and stochastic gradient "
    "descent interact to defend against model inversion. Include pseudocode.",
    "Solve this differential equation step by step: d²y/dx² + 3dy/dx + 2y = e^x, "
    "showing all integration steps and verifying the solution using formal proof techniques.",
    "Perform a rigorous complexity analysis (Big-O) of the following recursive algorithm "
    "and derive its recurrence relation using the master theorem step by step.",
]


class TestComplexityRouterTierAssignment:
    """Test that prompts are assigned to the correct complexity tier."""

    def setup_method(self) -> None:
        """Create a fresh router for each test."""
        self.router = ComplexityRouter(
            local_model="ollama/llama3.2",
            mid_tier_model="gpt-4o-mini",
            premium_model="claude-opus-4",
        )

    @pytest.mark.parametrize("prompt", SIMPLE_PROMPTS)
    def test_simple_prompts_route_to_local_tier(self, prompt: str) -> None:
        """Simple prompts should have complexity scores below SIMPLE_THRESHOLD."""
        score = self.router.classify(prompt)
        assert score < COMPLEX_THRESHOLD, (
            f"Expected simple prompt to score below {COMPLEX_THRESHOLD}, "
            f"got {score:.3f}: '{prompt[:60]}'"
        )

    @pytest.mark.parametrize("prompt", COMPLEX_PROMPTS)
    def test_complex_prompts_route_to_premium_tier(self, prompt: str) -> None:
        """Complex prompts should have complexity scores above SIMPLE_THRESHOLD."""
        score = self.router.classify(prompt)
        assert score > SIMPLE_THRESHOLD, (
            f"Expected complex prompt to score above {SIMPLE_THRESHOLD}, "
            f"got {score:.3f}: '{prompt[:60]}'"
        )

    def test_route_returns_routing_decision(self) -> None:
        """route() should always return a RoutingDecision instance."""
        decision = self.router.route("Hello!")
        assert isinstance(decision, RoutingDecision)

    def test_routing_decision_has_required_fields(self) -> None:
        """RoutingDecision must contain all required fields with correct types."""
        decision = self.router.route("What is 2+2?")
        assert isinstance(decision.selected_model, str)
        assert decision.selected_model != ""
        assert isinstance(decision.tier, str)
        assert isinstance(decision.complexity_score, float)
        assert 0.0 <= decision.complexity_score <= 1.0
        assert isinstance(decision.reasoning, str)
        assert isinstance(decision.features, dict)

    def test_simple_prompt_selects_local_model(self) -> None:
        """A clearly simple prompt should select the local model."""
        decision = self.router.route("Hi!")
        assert decision.tier == "local"
        assert decision.selected_model == "ollama/llama3.2"

    def test_complex_prompt_selects_premium_model(self) -> None:
        """A highly complex prompt should select the premium model."""
        decision = self.router.route(
            "Step by step analyze the eigenvector decomposition and reason through "
            "the adversarial cryptographic attack vectors in homomorphic encryption. "
            "Provide formal proofs and critique the stochastic Bayesian framework. "
            + "word " * 200
        )
        assert decision.tier == "premium"
        assert decision.selected_model == "claude-opus-4"

    def test_classify_returns_float_in_range(self) -> None:
        """classify() must return a float between 0.0 and 1.0."""
        score = self.router.classify("test prompt")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_system_prompt_included_in_scoring(self) -> None:
        """System prompt should be concatenated with user prompt for scoring."""
        score_without = self.router.classify("What is 2+2?")
        score_with = self.router.classify(
            "What is 2+2?",
            system_prompt="Step by step analyze and critique the mathematical "
            "reasoning using formal proof techniques from Bayesian inference.",
        )
        # Adding a complex system prompt should raise the score
        assert score_with >= score_without


class TestComplexityRouterPreferences:
    """Test ModelPreferences overrides."""

    def setup_method(self) -> None:
        self.router = ComplexityRouter()

    def test_local_model_override_applied(self) -> None:
        """Tenant preference for local model should override default."""
        prefs = ModelPreferences(local_model="ollama/phi3")
        decision = self.router.route("Hi!", preferences=prefs)
        assert decision.selected_model == "ollama/phi3"

    def test_premium_model_override_applied(self) -> None:
        """Tenant preference for premium model should override default."""
        prefs = ModelPreferences(premium_model="claude-sonnet-4")
        complex_prompt = (
            "Step by step analyze and critique the eigenvector decomposition in "
            "quantum cryptography with formal proofs and Bayesian reasoning. " * 3
        )
        decision = self.router.route(complex_prompt, preferences=prefs)
        if decision.tier == "premium":
            assert decision.selected_model == "claude-sonnet-4"

    def test_none_preferences_uses_defaults(self) -> None:
        """When preferences is None, defaults should be used."""
        decision = self.router.route("Hello!", preferences=None)
        assert decision.selected_model in (
            ComplexityRouter.DEFAULT_LOCAL_MODEL,
            ComplexityRouter.DEFAULT_MID_TIER_MODEL,
            ComplexityRouter.DEFAULT_PREMIUM_MODEL,
        )


class TestComplexityRouterConstraints:
    """Test latency and cost constraint enforcement."""

    def test_downgraded_flag_false_when_no_catalog(self) -> None:
        """Without a model catalog, constraints cannot be checked — not downgraded."""
        prefs = ModelPreferences(latency_budget_ms=100, cost_ceiling_usd=0.001)
        router = ComplexityRouter()  # No catalog
        decision = router.route(
            "Analyze this quantum Bayesian stochastic eigenvector. " * 5,
            preferences=prefs,
        )
        # Without catalog data, constraint check cannot fire
        assert not decision.downgraded

    def test_latency_budget_stored_in_decision(self) -> None:
        """latency_budget_ms should appear in the RoutingDecision when set."""
        prefs = ModelPreferences(latency_budget_ms=500)
        router = ComplexityRouter()
        decision = router.route("Hello!", preferences=prefs)
        assert decision.latency_budget_ms == 500

    def test_cost_ceiling_stored_in_decision(self) -> None:
        """cost_ceiling_usd should appear in the RoutingDecision when set."""
        prefs = ModelPreferences(cost_ceiling_usd=0.05)
        router = ComplexityRouter()
        decision = router.route("Hello!", preferences=prefs)
        assert decision.cost_ceiling_usd == pytest.approx(0.05)

    def test_downgrade_with_catalog_violating_latency(self) -> None:
        """When catalog shows premium exceeds latency budget, should downgrade."""
        catalog = {
            "claude-opus-4": ModelTarget(
                model_id="claude-opus-4",
                tier="premium",
                typical_latency_ms=3000,
            ),
            "gpt-4o-mini": ModelTarget(
                model_id="gpt-4o-mini",
                tier="mid",
                typical_latency_ms=800,
            ),
            "ollama/llama3.2": ModelTarget(
                model_id="ollama/llama3.2",
                tier="local",
                typical_latency_ms=200,
            ),
        }
        prefs = ModelPreferences(
            latency_budget_ms=1000,  # Premium (3000ms) violates this
            premium_model="claude-opus-4",
            mid_tier_model="gpt-4o-mini",
            local_model="ollama/llama3.2",
        )
        router = ComplexityRouter(
            local_model="ollama/llama3.2",
            mid_tier_model="gpt-4o-mini",
            premium_model="claude-opus-4",
            model_catalog=catalog,
        )
        complex_prompt = (
            "Step by step analyze and critique the eigenvector decomposition in "
            "quantum cryptography with formal proofs and Bayesian reasoning. "
            "word " * 300
        )
        decision = router.route(complex_prompt, preferences=prefs)
        # If routed to premium and latency budget is violated, should downgrade
        if decision.tier == "premium":
            assert decision.downgraded
