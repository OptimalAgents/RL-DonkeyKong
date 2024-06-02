#!/bin/bash

# dqn with all (ladder, level, stars, punish_needless_jump, heuristic_actions) with epsilon_end=0.04
python -m src/main --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 1000 \
    --training_starts 84 \
    --eval_every 10 \
    --punish_death \
    --heuristic_actions \
    --epsilon_end 0.04