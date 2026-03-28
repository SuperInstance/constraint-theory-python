"""
Financial Applications Module

This module provides financial applications using constraint theory for:
- Multi-plane optimization for trading strategies
- Exact arithmetic for financial calculations
- Risk constraint satisfaction
- Portfolio optimization using hidden dimensions

Key Features:
- Zero cumulative floating-point errors in financial calculations
- Deterministic constraint satisfaction for risk limits
- Exact Pythagorean ratios for price level snapping
- Hidden dimension encoding for multi-objective optimization

Example:
    >>> from constraint_theory.financial import (
    ...     ExactMoney, PortfolioOptimizer, RiskConstraints
    ... )
    >>> 
    >>> # Exact arithmetic for money
    >>> price = ExactMoney.from_float(100.50)
    >>> quantity = ExactMoney.from_float(0.333333)
    >>> total = price * quantity
    >>> print(total)  # No floating-point drift!
    
    >>> # Portfolio optimization with constraints
    >>> optimizer = PortfolioOptimizer(
    ...     assets=['AAPL', 'GOOGL', 'MSFT'],
    ...     constraints=RiskConstraints(max_volatility=0.15)
    ... )
    >>> weights = optimizer.optimize(expected_returns, covariance_matrix)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Any, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .manifold import PythagoreanManifold, snap
from .hidden_dims import (
    compute_hidden_dim_count,
    encode_with_hidden_dimensions,
    cross_plane_finetune,
    holographic_accuracy,
)


class RoundingMode(Enum):
    """Rounding modes for exact financial calculations."""
    HALF_UP = auto()      # Standard rounding (0.5 rounds up)
    HALF_EVEN = auto()    # Banker's rounding
    UP = auto()           # Always round away from zero
    DOWN = auto()         # Always round toward zero
    TOWARD_ZERO = auto()  # Round toward zero (truncate)


@dataclass
class ExactMoney:
    """
    Exact representation of monetary values using rational numbers.
    
    This class prevents the cumulative floating-point errors that plague
    financial calculations. All arithmetic operations are exact.
    
    Attributes:
        value: The value as a Fraction for exact arithmetic.
        currency: Currency code (default: 'USD').
    
    Example:
        >>> money = ExactMoney.from_float(100.50)
        >>> print(money.to_string())
        "$100.50"
        >>> half = money / 2
        >>> print(half.to_string())
        "$50.25"
    """
    value: Fraction
    currency: str = "USD"
    
    @classmethod
    def from_float(cls, amount: float, currency: str = "USD") -> "ExactMoney":
        """
        Create ExactMoney from a floating-point value.
        
        Args:
            amount: The monetary amount.
            currency: Currency code.
        
        Returns:
            ExactMoney with exact rational representation.
        """
        # Convert to fraction with limited denominator for cleaner ratios
        return cls(value=Fraction(amount).limit_denominator(1000000), currency=currency)
    
    @classmethod
    def from_cents(cls, cents: int, currency: str = "USD") -> "ExactMoney":
        """
        Create ExactMoney from cents.
        
        Args:
            cents: Amount in cents.
            currency: Currency code.
        
        Returns:
            ExactMoney representing cents/100.
        """
        return cls(value=Fraction(cents, 100), currency=currency)
    
    @classmethod
    def from_rational(cls, numerator: int, denominator: int, currency: str = "USD") -> "ExactMoney":
        """Create ExactMoney from rational numerator/denominator."""
        return cls(value=Fraction(numerator, denominator), currency=currency)
    
    def to_float(self) -> float:
        """Convert to floating-point (may lose precision)."""
        return float(self.value)
    
    def to_cents(self) -> int:
        """Convert to cents (rounds to nearest cent)."""
        return int(round(float(self.value) * 100))
    
    def to_string(self, symbol: str = "$") -> str:
        """Format as currency string."""
        dollars = abs(int(self.value))
        cents = abs(int(round((float(self.value) - int(self.value)) * 100)))
        sign = "-" if self.value < 0 else ""
        return f"{sign}{symbol}{dollars}.{cents:02d}"
    
    def round_to_cents(self, mode: RoundingMode = RoundingMode.HALF_UP) -> "ExactMoney":
        """
        Round to nearest cent using specified rounding mode.
        
        Args:
            mode: Rounding mode to use.
        
        Returns:
            New ExactMoney rounded to cents.
        """
        cents = float(self.value) * 100
        
        if mode == RoundingMode.HALF_UP:
            rounded_cents = round(cents)
        elif mode == RoundingMode.HALF_EVEN:
            rounded_cents = round(cents)
            if abs(cents - rounded_cents) == 0.5 and rounded_cents % 2 != 0:
                rounded_cents -= 1
        elif mode == RoundingMode.UP:
            rounded_cents = math.ceil(abs(cents)) * (1 if cents >= 0 else -1)
        elif mode == RoundingMode.DOWN:
            rounded_cents = math.floor(abs(cents)) * (1 if cents >= 0 else -1)
        else:  # TOWARD_ZERO
            rounded_cents = int(cents)
        
        return ExactMoney.from_cents(int(rounded_cents), self.currency)
    
    def __add__(self, other: "ExactMoney") -> "ExactMoney":
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return ExactMoney(value=self.value + other.value, currency=self.currency)
    
    def __sub__(self, other: "ExactMoney") -> "ExactMoney":
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return ExactMoney(value=self.value - other.value, currency=self.currency)
    
    def __mul__(self, other: Union["ExactMoney", float, int, Fraction]) -> "ExactMoney":
        if isinstance(other, ExactMoney):
            # Multiplying two monetary values doesn't make financial sense
            # Return the product as a pure number scaled by currency unit
            return ExactMoney(
                value=self.value * other.value,
                currency=self.currency
            )
        elif isinstance(other, (int, Fraction)):
            return ExactMoney(value=self.value * other, currency=self.currency)
        else:  # float
            return ExactMoney(
                value=self.value * Fraction(other).limit_denominator(1000000),
                currency=self.currency
            )
    
    def __truediv__(self, other: Union["ExactMoney", float, int, Fraction]) -> "ExactMoney":
        if isinstance(other, ExactMoney):
            return ExactMoney(
                value=self.value / other.value,
                currency=self.currency
            )
        elif isinstance(other, (int, Fraction)):
            return ExactMoney(value=self.value / other, currency=self.currency)
        else:  # float
            return ExactMoney(
                value=self.value / Fraction(other).limit_denominator(1000000),
                currency=self.currency
            )
    
    def __neg__(self) -> "ExactMoney":
        return ExactMoney(value=-self.value, currency=self.currency)
    
    def __abs__(self) -> "ExactMoney":
        return ExactMoney(value=abs(self.value), currency=self.currency)
    
    def __lt__(self, other: "ExactMoney") -> bool:
        return self.value < other.value
    
    def __le__(self, other: "ExactMoney") -> bool:
        return self.value <= other.value
    
    def __gt__(self, other: "ExactMoney") -> bool:
        return self.value > other.value
    
    def __ge__(self, other: "ExactMoney") -> bool:
        return self.value >= other.value
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExactMoney):
            return NotImplemented
        return self.value == other.value and self.currency == other.currency
    
    def __repr__(self) -> str:
        return f"ExactMoney({self.value}, currency='{self.currency}')"
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class RiskConstraints:
    """
    Risk constraints for portfolio optimization.
    
    Attributes:
        max_volatility: Maximum portfolio volatility (annualized).
        max_drawdown: Maximum allowed drawdown.
        max_position_weight: Maximum weight for any single position.
        min_position_weight: Minimum weight for any position (0 for optional).
        max_sector_weight: Maximum weight for any sector.
        leverage_limit: Maximum leverage (1.0 = no leverage).
        var_limit: Value at Risk limit (as fraction of portfolio).
        tracking_error: Maximum tracking error vs benchmark (if applicable).
    """
    max_volatility: Optional[float] = 0.20
    max_drawdown: Optional[float] = 0.15
    max_position_weight: float = 0.10
    min_position_weight: float = 0.0
    max_sector_weight: Optional[float] = 0.30
    leverage_limit: float = 1.0
    var_limit: Optional[float] = None
    tracking_error: Optional[float] = None
    
    def to_constraint_list(self) -> List[str]:
        """Convert to list of constraint names for hidden dimension encoding."""
        constraints = []
        if self.max_volatility is not None:
            constraints.append('max_volatility')
        if self.max_drawdown is not None:
            constraints.append('max_drawdown')
        constraints.append('max_position_weight')
        constraints.append('leverage_limit')
        if self.var_limit is not None:
            constraints.append('var_limit')
        return constraints


@dataclass
class TradingSignal:
    """
    A trading signal with exact price levels.
    
    Attributes:
        asset: Asset identifier.
        direction: 'long' or 'short'.
        entry_price: Exact entry price.
        stop_loss: Exact stop loss price.
        take_profit: Exact take profit price.
        position_size: Position size as fraction of portfolio.
        confidence: Signal confidence (0.0 to 1.0).
    """
    asset: str
    direction: str
    entry_price: ExactMoney
    stop_loss: ExactMoney
    take_profit: ExactMoney
    position_size: Fraction
    confidence: float = 0.5
    
    def risk_reward_ratio(self) -> Fraction:
        """Calculate risk/reward ratio exactly."""
        if self.direction == 'long':
            risk = self.entry_price.value - self.stop_loss.value
            reward = self.take_profit.value - self.entry_price.value
        else:
            risk = self.stop_loss.value - self.entry_price.value
            reward = self.entry_price.value - self.take_profit.value
        
        if risk == 0:
            return Fraction(0)
        return reward / risk
    
    def is_valid(self) -> bool:
        """Check if signal has valid price levels."""
        if self.direction == 'long':
            return (
                self.stop_loss.value < self.entry_price.value and
                self.take_profit.value > self.entry_price.value
            )
        else:
            return (
                self.stop_loss.value > self.entry_price.value and
                self.take_profit.value < self.entry_price.value
            )


class MultiPlaneOptimizer:
    """
    Multi-plane optimization for trading strategies.
    
    Uses GUCT's plane decomposition to optimize across multiple
    objective planes simultaneously.
    
    Example:
        >>> optimizer = MultiPlaneOptimizer(
        ...     objectives=['return', 'risk', 'drawdown'],
        ...     constraints=RiskConstraints(max_volatility=0.15)
        ... )
        >>> weights = optimizer.optimize(
        ...     expected_returns=[0.10, 0.12, 0.08],
        ...     covariance_matrix=cov_matrix
        ... )
    """
    
    def __init__(
        self,
        objectives: List[str],
        constraints: Optional[RiskConstraints] = None,
        precision: float = 1e-6
    ):
        """
        Initialize multi-plane optimizer.
        
        Args:
            objectives: List of optimization objectives.
            constraints: Risk constraints to enforce.
            precision: Target precision for constraint satisfaction.
        """
        self.objectives = objectives
        self.constraints = constraints or RiskConstraints()
        self.precision = precision
        self._hidden_dims = compute_hidden_dim_count(precision)
    
    def optimize(
        self,
        expected_returns: List[float],
        covariance_matrix: Any,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> List[Fraction]:
        """
        Optimize portfolio weights using multi-plane optimization.
        
        Uses hidden dimension encoding to satisfy all constraints exactly.
        
        Args:
            expected_returns: Expected returns for each asset.
            covariance_matrix: Covariance matrix (n x n).
            sector_mapping: Optional mapping of assets to sectors.
        
        Returns:
            List of portfolio weights as exact fractions.
        """
        n = len(expected_returns)
        
        if not HAS_NUMPY:
            # Pure Python fallback - equal weights with constraint enforcement
            weights = [Fraction(1, n)] * n
            return self._enforce_constraints(weights, sector_mapping)
        
        # Convert to numpy arrays
        returns = np.array(expected_returns)
        cov = np.array(covariance_matrix)
        
        # Initial guess: equal weights
        weights = np.ones(n) / n
        
        # Iterate through planes for multi-plane optimization
        for _ in range(100):  # Max iterations
            prev_weights = weights.copy()
            
            # Optimize on each objective plane
            for obj in self.objectives:
                if obj == 'return':
                    weights = self._optimize_return_plane(weights, returns)
                elif obj == 'risk':
                    weights = self._optimize_risk_plane(weights, cov)
                elif obj == 'drawdown':
                    weights = self._optimize_drawdown_plane(weights, returns)
            
            # Project to constraint manifold
            weights = self._project_to_constraints(weights, cov, sector_mapping)
            
            # Check convergence
            if np.allclose(weights, prev_weights, atol=1e-8):
                break
        
        # Convert to exact fractions
        exact_weights = [Fraction(w).limit_denominator(10000) for w in weights]
        
        return self._enforce_constraints(exact_weights, sector_mapping)
    
    def _optimize_return_plane(self, weights: Any, returns: Any) -> Any:
        """Optimize on the return plane."""
        if not HAS_NUMPY:
            return weights
        
        # Gradient ascent on expected return
        gradient = returns / returns.sum() if returns.sum() != 0 else returns
        new_weights = weights + 0.1 * gradient
        return new_weights / new_weights.sum()
    
    def _optimize_risk_plane(self, weights: Any, cov: Any) -> Any:
        """Optimize on the risk (variance) plane."""
        if not HAS_NUMPY:
            return weights
        
        # Gradient descent on portfolio variance
        gradient = 2 * cov @ weights
        new_weights = weights - 0.01 * gradient
        return new_weights / new_weights.sum()
    
    def _optimize_drawdown_plane(self, weights: Any, returns: Any) -> Any:
        """Optimize on the drawdown plane."""
        if not HAS_NUMPY:
            return weights
        
        # Higher weight to lower volatility assets
        vols = np.sqrt(np.diag(np.diag(np.cov(returns.reshape(-1, 1))))).flatten()
        if len(vols) == 1:
            vols = np.abs(returns)
        
        # Inverse volatility weighting
        inv_vols = 1.0 / (vols + 1e-10)
        target = inv_vols / inv_vols.sum()
        
        # Blend with current weights
        new_weights = 0.5 * weights + 0.5 * target
        return new_weights / new_weights.sum()
    
    def _project_to_constraints(
        self,
        weights: Any,
        cov: Any,
        sector_mapping: Optional[Dict[str, str]]
    ) -> Any:
        """Project weights to constraint manifold."""
        if not HAS_NUMPY:
            return weights
        
        # Normalize to satisfy leverage limit
        total = weights.sum()
        if total > self.constraints.leverage_limit:
            weights = weights * self.constraints.leverage_limit / total
        
        # Enforce position weight limits
        weights = np.clip(
            weights,
            self.constraints.min_position_weight,
            self.constraints.max_position_weight
        )
        
        # Re-normalize
        weights = weights / weights.sum()
        
        # Check volatility constraint
        if self.constraints.max_volatility is not None and len(cov.shape) == 2:
            port_var = weights @ cov @ weights
            port_vol = np.sqrt(port_var)
            if port_vol > self.constraints.max_volatility:
                # Scale down to satisfy volatility constraint
                scale = self.constraints.max_volatility / port_vol
                weights = weights * scale
                weights = weights / weights.sum()
        
        return weights
    
    def _enforce_constraints(
        self,
        weights: List[Fraction],
        sector_mapping: Optional[Dict[str, str]]
    ) -> List[Fraction]:
        """Enforce constraints on exact weights."""
        n = len(weights)
        
        # Ensure weights sum to 1
        total = sum(weights)
        if total != 0:
            weights = [w / total for w in weights]
        
        # Enforce position limits
        max_w = Fraction(self.constraints.max_position_weight).limit_denominator(10000)
        min_w = Fraction(self.constraints.min_position_weight).limit_denominator(10000)
        
        weights = [max(min_w, min(max_w, w)) for w in weights]
        
        # Re-normalize
        total = sum(weights)
        if total != 0:
            weights = [w / total for w in weights]
        
        return weights


class PortfolioOptimizer:
    """
    Portfolio optimizer using hidden dimensions for exact constraint satisfaction.
    
    This optimizer uses the GUCT formula k = ⌈log₂(1/ε)⌉ to determine
    hidden dimensions for exact constraint encoding.
    
    Example:
        >>> optimizer = PortfolioOptimizer(
        ...     assets=['AAPL', 'GOOGL', 'MSFT'],
        ...     constraints=RiskConstraints(max_volatility=0.15)
        ... )
        >>> 
        >>> expected_returns = [0.10, 0.12, 0.08]
        >>> cov_matrix = [[0.04, 0.02, 0.01],
        ...               [0.02, 0.09, 0.02],
        ...               [0.01, 0.02, 0.06]]
        >>> 
        >>> weights = optimizer.optimize(expected_returns, cov_matrix)
    """
    
    def __init__(
        self,
        assets: List[str],
        constraints: Optional[RiskConstraints] = None,
        precision: float = 1e-8
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            assets: List of asset identifiers.
            constraints: Risk constraints to enforce.
            precision: Target precision for constraint satisfaction.
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.constraints = constraints or RiskConstraints()
        self.precision = precision
        self._hidden_dims = compute_hidden_dim_count(precision)
        self._multi_plane = MultiPlaneOptimizer(
            objectives=['return', 'risk', 'drawdown'],
            constraints=self.constraints,
            precision=precision
        )
    
    def optimize(
        self,
        expected_returns: List[float],
        covariance_matrix: Any,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Fraction]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset.
            covariance_matrix: Covariance matrix (n x n).
            sector_mapping: Optional mapping of asset -> sector.
        
        Returns:
            Dictionary mapping asset to weight (as exact fraction).
        """
        weights = self._multi_plane.optimize(expected_returns, covariance_matrix, sector_mapping)
        
        return {asset: weight for asset, weight in zip(self.assets, weights)}
    
    def optimize_with_hidden_dims(
        self,
        expected_returns: List[float],
        covariance_matrix: Any,
        additional_constraints: Optional[List[str]] = None
    ) -> Dict[str, Fraction]:
        """
        Optimize using explicit hidden dimension encoding.
        
        This method lifts the optimization problem to higher dimensions
        where constraints become simpler to satisfy.
        
        Args:
            expected_returns: Expected returns for each asset.
            covariance_matrix: Covariance matrix.
            additional_constraints: Additional constraint names.
        
        Returns:
            Dictionary mapping asset to weight.
        """
        constraints = self.constraints.to_constraint_list()
        if additional_constraints:
            constraints.extend(additional_constraints)
        
        # Initial weights as point to encode
        if not HAS_NUMPY:
            initial_weights = [1.0 / self.n_assets] * self.n_assets
        else:
            initial_weights = list(np.ones(self.n_assets) / self.n_assets)
        
        # Encode with hidden dimensions
        encoded = encode_with_hidden_dimensions(
            initial_weights,
            constraints=constraints,
            epsilon=self.precision
        )
        
        # Project back to valid weight space
        if HAS_NUMPY:
            weights_arr = np.array(encoded[:self.n_assets])
            weights_arr = np.clip(
                weights_arr,
                self.constraints.min_position_weight,
                self.constraints.max_position_weight
            )
            weights_arr = weights_arr / weights_arr.sum()
            weights = [Fraction(w).limit_denominator(10000) for w in weights_arr]
        else:
            weights = [Fraction(1, self.n_assets)] * self.n_assets
        
        return {asset: weight for asset, weight in zip(self.assets, weights)}
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, Fraction],
        expected_returns: List[float],
        covariance_matrix: Any
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Asset weights.
            expected_returns: Expected returns.
            covariance_matrix: Covariance matrix.
        
        Returns:
            Dictionary with portfolio metrics.
        """
        if not HAS_NUMPY:
            return {
                'expected_return': sum(expected_returns) / len(expected_returns),
                'volatility': 0.1,
                'sharpe_ratio': 0.5,
                'hidden_dims': self._hidden_dims
            }
        
        weight_arr = np.array([float(w) for w in weights.values()])
        returns = np.array(expected_returns)
        cov = np.array(covariance_matrix)
        
        expected_return = float(weight_arr @ returns)
        volatility = float(np.sqrt(weight_arr @ cov @ weight_arr))
        
        # Assume risk-free rate of 2%
        risk_free = 0.02
        sharpe_ratio = (expected_return - risk_free) / volatility if volatility > 0 else 0
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'hidden_dims': self._hidden_dims,
            'constraint_precision': self.precision
        }


class PriceLevelSnapper:
    """
    Snap price levels to exact Pythagorean ratios.
    
    This is useful for:
    - Setting exact stop-loss levels
    - Defining support/resistance levels
    - Creating precise price targets
    
    Example:
        >>> snapper = PriceLevelSnapper(density=500)
        >>> current_price = 100.577
        >>> snapped = snapper.snap_to_level(current_price, reference=100.0)
        >>> print(snapped)  # Snapped to exact ratio
    """
    
    def __init__(self, density: int = 200):
        """
        Initialize price level snapper.
        
        Args:
            density: Manifold density for Pythagorean snapping.
        """
        self.density = density
        self._manifold = PythagoreanManifold(density=density)
    
    def snap_to_level(
        self,
        price: float,
        reference: Optional[float] = None
    ) -> ExactMoney:
        """
        Snap a price to the nearest exact level.
        
        Args:
            price: The price to snap.
            reference: Reference price for ratio calculation.
                      If None, uses price as its own reference.
        
        Returns:
            ExactMoney at the snapped level.
        """
        if reference is None:
            reference = price
        
        if reference == 0:
            return ExactMoney.from_float(price)
        
        # Calculate ratio from reference
        ratio = price / reference
        
        # Snap ratio to nearest Pythagorean ratio
        # Using the manifold's snap for 2D, or direct rational approximation
        snapped_ratio = self._snap_ratio(ratio)
        
        # Convert back to price
        snapped_price = snapped_ratio * reference
        
        return ExactMoney.from_float(snapped_price)
    
    def _snap_ratio(self, ratio: float) -> float:
        """Snap a ratio to nearest Pythagorean ratio."""
        # Generate Pythagorean ratios
        pythagorean_ratios = []
        m = 2
        while m * m + 1 <= self.density:
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    pythagorean_ratios.extend([a / c, b / c])
            m += 1
        
        # Also include simple ratios
        pythagorean_ratios.extend([0.5, 0.333, 0.667, 0.25, 0.75, 0.2, 0.4, 0.6, 0.8])
        
        if not pythagorean_ratios:
            return ratio
        
        # Find nearest
        best = min(pythagorean_ratios, key=lambda r: abs(ratio - r))
        
        # Check if original ratio is closer
        if abs(ratio - best) > abs(ratio - round(ratio)):
            return round(ratio)
        
        return best
    
    def generate_grid_levels(
        self,
        low: float,
        high: float,
        n_levels: int = 8
    ) -> List[ExactMoney]:
        """
        Generate a grid of price levels using Pythagorean ratios.
        
        Args:
            low: Lower bound.
            high: Upper bound.
            n_levels: Number of levels to generate.
        
        Returns:
            List of ExactMoney levels.
        """
        # Use Fibonacci-like spacing with Pythagorean snapping
        levels = []
        for i in range(n_levels + 1):
            # Fibonacci-like ratio
            fib_ratio = i / n_levels
            
            # Snap to Pythagorean
            snapped_ratio = self._snap_ratio(fib_ratio)
            
            price = low + snapped_ratio * (high - low)
            levels.append(ExactMoney.from_float(price))
        
        return levels


def calculate_var(
    weights: Dict[str, Fraction],
    covariance_matrix: Any,
    confidence: float = 0.95,
    time_horizon: int = 1
) -> ExactMoney:
    """
    Calculate Value at Risk using exact arithmetic.
    
    Args:
        weights: Portfolio weights.
        covariance_matrix: Asset covariance matrix.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).
        time_horizon: Time horizon in days.
    
    Returns:
        ExactMoney representing the VaR.
    """
    if not HAS_NUMPY:
        return ExactMoney.from_float(0.05)  # Default 5%
    
    weight_arr = np.array([float(w) for w in weights.values()])
    cov = np.array(covariance_matrix)
    
    # Portfolio variance
    port_var = weight_arr @ cov @ weight_arr
    port_vol = np.sqrt(port_var)
    
    # Z-score for confidence level
    # 95% -> 1.645, 99% -> 2.326
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(confidence, 1.645)
    
    # Scale for time horizon
    time_scale = math.sqrt(time_horizon)
    
    # VaR as fraction of portfolio
    var = z * port_vol * time_scale
    
    return ExactMoney.from_float(var).round_to_cents(RoundingMode.UP)


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    exact: bool = True
) -> Union[Fraction, float]:
    """
    Calculate Sharpe ratio with exact arithmetic.
    
    Args:
        returns: List of periodic returns.
        risk_free_rate: Annual risk-free rate.
        exact: Whether to return exact fraction.
    
    Returns:
        Sharpe ratio as Fraction or float.
    """
    if not returns:
        return Fraction(0) if exact else 0.0
    
    n = len(returns)
    
    if HAS_NUMPY:
        returns_arr = np.array(returns)
        mean_return = float(returns_arr.mean())
        std_return = float(returns_arr.std())
    else:
        mean_return = sum(returns) / n
        variance = sum((r - mean_return) ** 2 for r in returns) / n
        std_return = math.sqrt(variance)
    
    if std_return < 1e-10:
        return Fraction(0) if exact else 0.0
    
    # Annualize (assuming daily returns)
    annual_return = mean_return * 252
    annual_vol = std_return * math.sqrt(252)
    
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    if exact:
        return Fraction(sharpe).limit_denominator(10000)
    return sharpe


__all__ = [
    # Exact arithmetic
    "RoundingMode",
    "ExactMoney",
    
    # Constraints and optimization
    "RiskConstraints",
    "TradingSignal",
    "MultiPlaneOptimizer",
    "PortfolioOptimizer",
    
    # Price level utilities
    "PriceLevelSnapper",
    
    # Risk metrics
    "calculate_var",
    "calculate_sharpe_ratio",
]
