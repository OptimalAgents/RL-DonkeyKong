#!/bin/bash
sleep $((10 * $1))

# dqn with punish_needless_jump +ladder+level
python -m main --agent dqn \
    --punish_needless_jump \
    --level_incentive \
    --ladder_incentive \
    --total_steps 20000 \
    --training_starts 1000 \
    --eval_every 1000 \
    --punish_death