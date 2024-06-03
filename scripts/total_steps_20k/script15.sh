#!/bin/bash

# dqn with all (ladder, level, stars, punish_needless_jump, heuristic_actions) without punish_death
python -m main --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 20000 \
    --training_starts 1000 \
    --eval_every 1000 \
    --heuristic_actions