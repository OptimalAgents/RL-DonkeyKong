from typing import TypeAlias
import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm


BARREL_COLOR = np.array((236, 200, 96))
TORCH_MASK = np.zeros((210, 160))
TORCH_MASK[68:75, 40:42] = 1

MARIO_COLOR = np.array((200, 72, 72))

Point: TypeAlias = tuple[int, int]


def find_barrels(observation: np.ndarray) -> list[Point]:
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


def find_mario(observation: np.ndarray) -> Point:
    mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
    xs, ys = np.where(mario_mask)
    return int(np.mean(xs)), int(np.mean(ys))


def closest_barrel_dist(mario: Point, barrels: list[Point]) -> int:
    if not barrels:
        return 50

    mario_arr = np.array(mario)
    min_barrel = np.array(barrels[0])
    min_dist = np.linalg.norm(mario_arr - min_barrel)
    for barrel in barrels:
        barrel_arr = np.array(barrel)
        dist = np.linalg.norm(mario_arr - barrel_arr)
        if dist < min_dist:
            min_barrel = barrel
            min_dist = dist

    distance = int(min_dist)
    return min(distance, 50)


EPSILON = 0.1
LEARNING_RATE = 1
GAMMA = 0.1


def main():
    env = gym.make("ALE/DonkeyKong-v5", render_mode="rgb_array")
    # Mario x, matrio y, closes barrel dist, action
    q_table = np.zeros((210, 160, 51, 18), dtype=np.float32)

    # Training
    for epoch in tqdm(range(2)):
        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(1)

        mario = find_mario(observation)
        start_position = mario
        barrels = find_barrels(observation)
        barrel_dist = closest_barrel_dist(mario, barrels)
        state = [mario[0], mario[1], barrel_dist]

        reward_sum = 0
        for episode in range(3000):
            # if np.random.rand() < EPSILON:
            # else:
            #     action = np.argmax(q_table[mario[0], mario[1], barrel_dist])
            action = np.random.randint(18)

            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                reward = -100

            if mario[1] - start_position[1] > 8:
                reward = 10
                start_position = mario

            if mario[0] - start_position[0] < -8:
                reward = 100
                start_position = mario

            mario = find_mario(observation)
            barrels = find_barrels(observation)
            barrel_dist = closest_barrel_dist(mario, barrels)
            new_state = [mario[0], mario[1], barrel_dist]

            reward_sum += reward
            update = (
                reward + GAMMA * np.max(q_table[new_state, :]) - q_table[state, action]
            )
            q_table[state, action] = q_table[state, action] + LEARNING_RATE * update
            state = new_state

            if terminated:
                break

        print(f"Epoch: {epoch}, Reward: {reward_sum}")

    # Visualization
    print(q_table[np.where(q_table != 0)])
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(1)
    for i in range(3000):
        mario = find_mario(observation)
        start_position = mario
        barrels = find_barrels(observation)
        barrel_dist = closest_barrel_dist(mario, barrels)
        search_space = q_table[mario[0], mario[1], barrel_dist]
        action = np.argmax(search_space)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break

    env.close()


if __name__ == "__main__":
    main()
