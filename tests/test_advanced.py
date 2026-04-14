"""Tests for holonomy, cohomology, curvature, and percolation."""

import pytest
from constraint_theory_python import (
    HolonomyChecker,
    HolonomyResult,
    compute_holonomy,
    FastCohomology,
    CohomologyResult,
    RicciFlow,
    ricci_flow_step,
    FastPercolation,
    RigidityResult,
)


class TestHolonomy:
    def test_closed_cycle_is_identity(self):
        result = compute_holonomy([0.1, 0.2, -0.3])
        assert result.is_identity

    def test_checker_incremental(self):
        checker = HolonomyChecker()
        checker.apply(0.5)
        checker.apply(-0.5)
        result = checker.check_closed()
        assert result.is_identity

    def test_reset(self):
        checker = HolonomyChecker()
        checker.apply(1.0)
        checker.reset()
        result = checker.check_partial()
        assert result.is_identity


class TestCohomology:
    def test_triangle(self):
        cohom = FastCohomology()
        result = cohom.compute(vertices=3, edges=3, components=1)
        assert result.h0 == 1
        assert result.h1 == 1

    def test_tree(self):
        cohom = FastCohomology()
        result = cohom.compute(vertices=5, edges=4, components=1)
        assert result.h0 == 1
        assert result.h1 == 0

    def test_disconnected(self):
        cohom = FastCohomology()
        result = cohom.compute(vertices=4, edges=2, components=2)
        assert result.h0 == 2


class TestRicciFlow:
    def test_evolve_toward_zero(self):
        flow = RicciFlow(alpha=0.1, target=0.0)
        curvatures = [1.0, 0.5, -0.3]
        evolved = flow.evolve(curvatures)
        for e, c in zip(evolved, curvatures):
            assert abs(e) < abs(c) or c == 0.0

    def test_convenience_function(self):
        result = ricci_flow_step([1.0], alpha=0.5, target=0.0)
        assert result[0] == 0.5


class TestPercolation:
    def test_minimally_rigid(self):
        perc = FastPercolation()
        perc.add_edge(0, 1)
        perc.add_edge(0, 2)
        perc.add_edge(1, 2)
        result = perc.compute_rigidity(3)
        assert result.is_rigid

    def test_not_rigid(self):
        perc = FastPercolation()
        perc.add_edge(0, 1)
        result = perc.compute_rigidity(3)
        assert not result.is_rigid
