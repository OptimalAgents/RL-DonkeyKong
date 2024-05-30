import numpy as np
from typing import List, Tuple

BARREL_COLOR = np.array((236, 200, 96))
TORCH_MASK = np.zeros((210, 160))
TORCH_MASK[68:75, 40:42] = 1

MARIO_COLOR = np.array((200, 72, 72))


def find_barrels(observation: np.ndarray) -> List[Tuple[int, int]]:
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
        barrels.append((int(np.mean(xs)), int(np.mean(ys))))

        barrels_mask = np.logical_and(barrels_mask, np.logical_not(current_barrel))

    return barrels


def find_mario(observation: np.ndarray) -> Tuple[int, int]:
    mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
    xs, ys = np.where(mario_mask)
    return int(np.mean(xs)), int(np.mean(ys))


def is_barrel_near_mario(
    mario: Tuple[int, int], barrels: list[Tuple[int, int]]
) -> bool:  # mario 16x12 pixels, barrels 8x8
    return any(
        mario[0] + 8 > barrel[0] > mario[0] - 1
        and mario[1] + 16 > barrel[1] > mario[1] - 16
        for barrel in barrels
    )


LADDER_MASK = np.zeros((210, 160))
LADDER_MASK[160:196, 106:112] = 1  # First level
LADDER_MASK[128:169, 79:84] = 1  # Second level
LADDER_MASK[132:160, 47:53] = 1  # Second level
LADDER_MASK[102:140, 87:92] = 1  # Third level
LADDER_MASK[104:137, 108:112] = 1  # Third level
LADDER_MASK[74:112, 67:72] = 1  # Fourth level
LADDER_MASK[71:110, 48:52] = 1  # Fourth level
LADDER_MASK[45:82, 107:112] = 1  # Fifth level
LADDER_MASK[20:60, 76:80] = 1  # Sixth level
LADDER_MASK[20:60, 32:36] = 1  # Sixth level


def is_mario_on_ladder(mario: Tuple[int, int]) -> bool:
    if LADDER_MASK[mario[0], mario[1]] == 1:
        return True
    return False


ZERO_LEVEL_Y = 178
FIRST_LEVEL_Y = 158
SECOND_LEVEL_Y = 131
THIRD_LEVEL_Y = 104
FOURTH_LEVEL_Y = 77
FIFTH_LEVEL_Y = 50

LEVEL_ARRAY = np.ones(ZERO_LEVEL_Y + 1) * 5  # Start all at 5
LEVEL_ARRAY[FIFTH_LEVEL_Y + 1 :] = 4  # Below fifth level is 4
LEVEL_ARRAY[FOURTH_LEVEL_Y + 1 :] = 3  # etc
LEVEL_ARRAY[THIRD_LEVEL_Y + 1 :] = 2
LEVEL_ARRAY[SECOND_LEVEL_Y + 1 :] = 1
LEVEL_ARRAY[FIRST_LEVEL_Y + 1 :] = 0


def get_level(mario_y: int) -> int:
    return LEVEL_ARRAY[mario_y]
