#!/bin/bash
sleep $((10 * $1))

# dqn with stars, level and ladder_incentive
python main.py --agent dqn \
    --stars_incentive \
    --level_incentive \
    --ladder_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death