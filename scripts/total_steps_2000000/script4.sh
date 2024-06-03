#!/bin/bash

# dqn with stars incentive
python -m src/main --agent dqn \
    --stars_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death