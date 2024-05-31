from __future__ import annotations


def train_sarsa_agent(env, agent, num_episodes=1000, max_steps_per_episode=1000):
    env = ActionProbabilityModifier(
        env, agent
    )  # Use the modified environment for training

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_observation(observation)
        action = env.choose_action(state, observation)  # Initial action
        total_reward = 0

        for step in range(max_steps_per_episode):
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_observation)
            next_action = env.choose_action(next_state, next_observation)  # Next action
            agent.update_q(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward

            env.render()  # Render each step to display the game

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    return agent
