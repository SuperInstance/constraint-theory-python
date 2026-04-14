"""Tests for the quantizer module."""

import pytest
from constraint_theory_python import PythagoreanQuantizer, QuantizationMode, QuantizationResult, Rational


class TestRational:
    def test_value(self):
        r = Rational(3, 5)
        assert abs(r.value - 0.6) < 1e-10

    def test_zero_den_raises(self):
        with pytest.raises(ZeroDivisionError):
            Rational(1, 0)

    def test_pythagorean(self):
        r = Rational(3, 4)
        assert r.is_pythagorean()  # 3^2 + 4^2 = 5^2

    def test_non_pythagorean(self):
        r = Rational(2, 3)
        assert not r.is_pythagorean()


class TestQuantizer:
    def test_ternary(self):
        q = PythagoreanQuantizer(QuantizationMode.TERNARY)
        result = q.quantize([0.6, 0.8, -0.1, 0.0, 0.5])
        assert result.data == [1.0, 1.0, 0.0, 0.0, 1.0]
        assert result.mse >= 0

    def test_empty_vector(self):
        q = PythagoreanQuantizer(QuantizationMode.TERNARY)
        result = q.quantize([])
        assert result.data == []
        assert result.mse == 0.0

    def test_for_embeddings(self):
        q = PythagoreanQuantizer.for_embeddings()
        assert q.mode == QuantizationMode.POLAR

    def test_for_weights(self):
        q = PythagoreanQuantizer.for_weights()
        assert q.mode == QuantizationMode.TERNARY

    def test_turbo(self):
        q = PythagoreanQuantizer(QuantizationMode.TURBO, bits=4)
        result = q.quantize([0.1, 0.2, 0.3, 0.4])
        assert len(result.data) == 4
        assert result.mse >= 0

    def test_hybrid_unit_norm_selects_polar(self):
        q = PythagoreanQuantizer(QuantizationMode.HYBRID)
        result = q.quantize([0.6, 0.8])
        assert result.unit_norm_preserved

    def test_batch(self):
        q = PythagoreanQuantizer(QuantizationMode.TERNARY)
        results = q.quantize_batch([[0.6, 0.8], [1.0, -1.0]])
        assert len(results) == 2
