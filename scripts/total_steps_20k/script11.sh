#!/bin/bash

# dqn with heuristic_actions
python -m main --agent dqn \
    --total_steps 20000 \
    --training_starts 1000 \
    --eval_every 1000 \
    --punish_death \
    --heuristic_actions