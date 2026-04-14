"""Tests for the PythagoreanManifold and snap function."""

import math
import pytest

from constraint_theory_python import PythagoreanManifold, PythagoreanTriple, snap
from constraint_theory_python.errors import (
    CTError,
    NaNInputError,
    ZeroVectorError,
    InfinityInputError,
)


class TestPythagoreanTriple:
    def test_3_4_5(self):
        t = PythagoreanTriple(3, 4, 5)
        assert t.a ** 2 + t.b ** 2 == t.c ** 2
        nx, ny = t.normalized()
        assert abs(nx - 0.6) < 1e-10
        assert abs(ny - 0.8) < 1e-10

    def test_5_12_13(self):
        t = PythagoreanTriple(5, 12, 13)
        assert t.a ** 2 + t.b ** 2 == t.c ** 2

    def test_frozen(self):
        t = PythagoreanTriple(3, 4, 5)
        with pytest.raises(AttributeError):
            t.a = 99


class TestPythagoreanManifold:
    def test_construction(self):
        m = PythagoreanManifold(200)
        assert m.density == 200
        assert m.triple_count > 0
        assert m.state_count > 0
        assert repr(m)  # Should not raise

    def test_snap_exact_triple(self):
        m = PythagoreanManifold(200)
        exact, noise = snap(m, (0.6, 0.8))
        assert noise < 0.001
        assert abs(exact[0] - 0.6) < 0.01
        assert abs(exact[1] - 0.8) < 0.01

    def test_snap_method(self):
        m = PythagoreanManifold(200)
        exact, noise = m.snap((0.8, 0.6))
        assert noise < 0.01
        mag_sq = exact[0] ** 2 + exact[1] ** 2
        assert abs(mag_sq - 1.0) < 1e-6, f"Expected unit norm, got {mag_sq}"

    def test_snap_unit_norm(self):
        m = PythagoreanManifold(200)
        for angle in [i * 0.1 for i in range(1, 62)]:
            vec = (math.cos(angle), math.sin(angle))
            exact, noise = m.snap(vec)
            mag_sq = exact[0] ** 2 + exact[1] ** 2
            assert abs(mag_sq - 1.0) < 1e-6, f"Non-unit at angle {angle}: {mag_sq}"

    def test_snap_zero_vector_raises(self):
        m = PythagoreanManifold(50)
        with pytest.raises(ZeroVectorError):
            m.snap((0.0, 0.0))

    def test_snap_nan_raises(self):
        m = PythagoreanManifold(50)
        with pytest.raises(NaNInputError):
            m.snap((float("nan"), 0.0))

    def test_snap_inf_raises(self):
        m = PythagoreanManifold(50)
        with pytest.raises(InfinityInputError):
            m.snap((float("inf"), 0.0))

    def test_snap_batch(self):
        m = PythagoreanManifold(100)
        vectors = [(0.6, 0.8), (0.8, 0.6), (1.0, 0.0)]
        results = m.snap_batch(vectors)
        assert len(results) == 3
        for exact, noise in results:
            mag_sq = exact[0] ** 2 + exact[1] ** 2
            assert abs(mag_sq - 1.0) < 1e-6

    def test_invalid_density(self):
        with pytest.raises(ValueError):
            PythagoreanManifold(-1)

    def test_low_density(self):
        m = PythagoreanManifold(10)
        assert m.triple_count > 0
