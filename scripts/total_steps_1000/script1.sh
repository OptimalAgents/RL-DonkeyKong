#!/bin/bash

# dqn with punish_death
python -m src/main --agent dqn \
    --total_steps 1000 \
    --training_starts 84 \
    --eval_every 10 \
    --punish_death