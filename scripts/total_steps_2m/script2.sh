#!/bin/bash
sleep $((10 * $1))

# dqn with level_incentive
python main.py --agent dqn \
    --level_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death