from triton import Config


# Complete author-visible programs and physical bindings for the dense examples.
# No independent domains are multiplied to manufacture a search space.
AFFINE_CONFIGS = (
    Config({"B": 64}, backend_options={"lmul_eighths": 8}),
    Config({"B": 128}, backend_options={"lmul_eighths": 16}),
    Config({"B": 64}, backend_options={"lmul_eighths": 8, "unroll": 2}),
)

STATE_CONFIGS = (
    Config({"B": 16}, backend_options={"lmul_eighths": 8}),
    Config({"B": 32}, backend_options={"lmul_eighths": 8}),
    Config({"B": 64}, backend_options={"lmul_eighths": 16}),
)
