import numpy as np


def preprocess_observation(observation: np.ndarray, max_states: int = 1000) -> int:
    # Downsample by a factor of 8
    downsampled = observation[::8, ::8]
    # Flatten and hash the values to create a unique state index
    hashed = (
        hash(downsampled.tobytes()) % max_states
    )  # Hash and modulo to keep within range
    return hashed
