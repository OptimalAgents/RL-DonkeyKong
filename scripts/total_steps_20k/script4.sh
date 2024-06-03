#!/bin/bash

# dqn with stars incentive
python -m main --agent dqn \
    --stars_incentive \
    --total_steps 20000 \
    --training_starts 1000 \
    --eval_every 1000 \
    --punish_death