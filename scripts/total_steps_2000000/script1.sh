#!/bin/bash

# dqn with punish_death
python -m src/main --agent dqn \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death