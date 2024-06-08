#!/bin/bash
sleep $((10 * $1))

# dqn with all (ladder, level, stars, punish_needless_jump, heuristic_actions) with gamma=0.9
python main.py --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death \
    --heuristic_actions \
    --gamma 0.9