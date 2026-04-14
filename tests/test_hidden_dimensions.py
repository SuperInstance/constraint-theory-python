"""Tests for hidden dimension encoding."""

import pytest
from constraint_theory_python import hidden_dim_count, lift_to_hidden, project_to_visible


class TestHiddenDimensions:
    def test_basic_counts(self):
        assert hidden_dim_count(0.1) == 4
        assert hidden_dim_count(0.01) == 7
        assert hidden_dim_count(0.001) == 10
        assert hidden_dim_count(1e-10) == 34

    def test_zero_epsilon(self):
        assert hidden_dim_count(0.0) == float("inf")

    def test_negative_epsilon(self):
        assert hidden_dim_count(-1.0) == float("inf")

    def test_lift_to_hidden(self):
        point = [0.6, 0.8]
        k = 34
        lifted = lift_to_hidden(point, k)
        assert len(lifted) == 36  # 2 visible + 34 hidden

    def test_project_to_visible(self):
        point = [0.6, 0.8]
        lifted = lift_to_hidden(point, 10)
        projected = project_to_visible(lifted, 2)
        assert projected == point
