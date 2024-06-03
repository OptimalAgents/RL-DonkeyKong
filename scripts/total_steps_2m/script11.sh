#!/bin/bash
sleep $((10 * $1))

# dqn with heuristic_actions
python -m src/main --agent dqn \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death \
    --heuristic_actions