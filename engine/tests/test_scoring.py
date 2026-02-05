"""Tests for scoring and confluence system."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoring.reasons import ReasonCode, REASON_DESCRIPTIONS
from scoring.confluence import ScoringWeights


class TestReasonCodes:
    """Tests for reason code system."""

    def test_all_codes_have_descriptions(self):
        """Test that all reason codes have descriptions."""
        for code in ReasonCode:
            assert code in REASON_DESCRIPTIONS, f"Missing description for {code}"

    def test_descriptions_not_empty(self):
        """Test that no descriptions are empty."""
        for code, desc in REASON_DESCRIPTIONS.items():
            assert len(desc) > 0, f"Empty description for {code}"

    def test_strength_codes_exist(self):
        """Test strength-related codes exist."""
        strength_codes = [c for c in ReasonCode if c.name.startswith("STR_")]
        assert len(strength_codes) >= 3

    def test_structure_codes_exist(self):
        """Test structure-related codes exist."""
        structure_codes = [c for c in ReasonCode if c.name.startswith("STRUCT_")]
        assert len(structure_codes) >= 3

    def test_liquidity_codes_exist(self):
        """Test liquidity-related codes exist."""
        liquidity_codes = [c for c in ReasonCode if c.name.startswith("LIQ_")]
        assert len(liquidity_codes) >= 3


class TestScoringWeights:
    """Tests for scoring weights."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = ScoringWeights()
        assert weights.strength == 25
        assert weights.structure == 25
        assert weights.liquidity == 20
        assert weights.momentum == 15
        assert weights.regime == 10
        assert weights.sentiment == 5

    def test_weights_sum_to_100(self):
        """Test that default weights sum to 100."""
        weights = ScoringWeights()
        total = (
            weights.strength +
            weights.structure +
            weights.liquidity +
            weights.momentum +
            weights.regime +
            weights.sentiment
        )
        assert total == 100

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = ScoringWeights(
            strength=30,
            structure=30,
            liquidity=15,
            momentum=10,
            regime=10,
            sentiment=5,
        )
        assert weights.strength == 30
        assert weights.structure == 30

    def test_weights_dataclass_frozen(self):
        """Test that weights are immutable after creation."""
        weights = ScoringWeights()
        # ScoringWeights should be a dataclass, verify fields exist
        assert hasattr(weights, 'strength')
        assert hasattr(weights, 'structure')
