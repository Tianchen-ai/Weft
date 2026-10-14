from weft.runtime import Config


# Complete author-visible programs and physical bindings for the dense examples.
# No independent domains are multiplied to manufacture a search space.
AFFINE_CONFIGS = (
    Config(name="block64-m1", meta={"B": 64}, lmul_eighths=8),
    Config(name="block128-m2", meta={"B": 128}, lmul_eighths=16),
    Config(name="block64-unroll2", meta={"B": 64}, lmul_eighths=8, unroll=2),
)

STATE_CONFIGS = (
    Config(name="block16-m1", meta={"B": 16}, lmul_eighths=8),
    Config(name="block32-m1", meta={"B": 32}, lmul_eighths=8),
    Config(name="block64-m2", meta={"B": 64}, lmul_eighths=16),
)
