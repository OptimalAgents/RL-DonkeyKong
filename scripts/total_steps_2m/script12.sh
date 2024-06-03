#!/bin/bash

# dqn with all (ladder, level, stars, punish_needless_jump, heuristic_actions) and more total_steps
python -m src/main --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 4000000 \
    --training_starts 80000 \
    --punish_death \
    --heuristic_actions