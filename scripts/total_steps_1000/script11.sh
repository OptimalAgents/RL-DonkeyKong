#!/bin/bash

# dqn with heuristic_actions
python -m src/main --agent dqn \
    --total_steps 1000 \
    --training_starts 84 \
    --eval_every 10 \
    --punish_death \
    --heuristic_actions