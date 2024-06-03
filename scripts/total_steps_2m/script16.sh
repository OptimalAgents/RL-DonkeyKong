#!/bin/bash
sleep $((10 * $1))

# q_learning with all (ladder, level, stars, punish_needless_jump, heuristic_actions)
python -m src/main --agent q_learning \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --stars_incentive \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death \
    --heuristic_actions