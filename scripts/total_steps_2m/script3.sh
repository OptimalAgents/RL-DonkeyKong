#!/bin/bash
sleep $((10 * $1))

# dqn with ladder and level_incentive
python -m src/main --agent dqn \
    --level_incentive \
    --ladder_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death