"""Example: Quantization modes for different use cases."""

from constraint_theory_python import PythagoreanQuantizer, QuantizationMode

# Ternary (BitNet) — for LLM weights
ternary = PythagoreanQuantizer(QuantizationMode.TERNARY)
weights = [0.6, -0.3, 0.01, -0.8, 0.0, 0.5]
result = ternary.quantize(weights)
print(f"Ternary: {result.data}")
print(f"MSE: {result.mse:.4f}")

# Polar — for embeddings (preserves unit norm)
polar = PythagoreanQuantizer.for_embeddings()
embedding = [0.6, 0.8]
result = polar.quantize(embedding)
print(f"\nPolar: {result.data}")
print(f"Unit norm: {result.check_unit_norm(0.1)}")

# Turbo — for vector databases
turbo = PythagoreanQuantizer(QuantizationMode.TURBO, bits=4)
vector = [0.1, 0.2, 0.3, 0.4, 0.5]
result = turbo.quantize(vector)
print(f"\nTurbo (4-bit): {result.data}")
print(f"MSE: {result.mse:.6f}")

# Hybrid — auto-select
hybrid = PythagoreanQuantizer(QuantizationMode.HYBRID)
for label, vec in [("embedding", [0.6, 0.8]), ("sparse", [0.5, 0.0, 0.0, -0.3])]:
    result = hybrid.quantize(vec)
    print(f"\nHybrid '{label}': {result.data} (unit_norm={result.unit_norm_preserved})")
