#!/bin/bash

# dqn with all (ladder, level, stars, punish_needless_jump, heuristic_actions) without punish_death
python -m src/main --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --heuristic_actions