#!/bin/bash
sleep $((10 * $1))

# sarsa with all (ladder, level, stars, punish_needless_jump, heuristic_actions)
python -m src/main --agent sarsa \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death \
    --heuristic_actions