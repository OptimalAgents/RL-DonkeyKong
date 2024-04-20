import numpy as np

BARREL_COLOR = np.array((236, 200, 96))
TORCH_MASK = np.zeros((210, 160))
TORCH_MASK[68:75, 40:42] = 1

MARIO_COLOR = np.array((200, 72, 72))


def find_barrels(observation: np.ndarray) -> list[tuple[int, int]]:
    barrels = []
    barrels_mask = np.all(observation == BARREL_COLOR, axis=-1)
    barrels_mask = np.logical_and(barrels_mask, np.logical_not(TORCH_MASK))
    while np.any(barrels_mask):
        xs, ys = np.where(barrels_mask)
        x, y = xs[0], ys[0]

        x_start, x_end = max(x - 8, 0), min(x + 8, 210)
        y_start, y_end = max(y - 8, 0), min(y + 8, 160)
        search_space = np.zeros((210, 160))
        search_space[x_start : (x_end + 1), y_start : (y_end + 1)] = 1

        current_barrel = np.logical_and(barrels_mask, search_space)

        xs, ys = np.where(current_barrel)
        barrels.append((int(np.mean(xs[0])), int(np.mean(ys[0]))))
        barrels_mask = np.logical_and(barrels_mask, np.logical_not(current_barrel))

    return barrels


def find_mario(observation: np.ndarray) -> tuple[int, int]:
    mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
    xs, ys = np.where(mario_mask)
    return int(np.mean(xs)), int(np.mean(ys))
