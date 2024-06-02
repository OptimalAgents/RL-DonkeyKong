#!/bin/bash

# dqn with stars incentive
python -m src/main --agent dqn \
    --stars_incentive \
    --total_steps 1000 \
    --training_starts 84 \
    --eval_every 10 \
    --punish_death