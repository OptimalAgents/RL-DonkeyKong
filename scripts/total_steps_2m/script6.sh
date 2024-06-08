#!/bin/bash
sleep $((10 * $1))

# dqn with punish_needless_jump
python main.py --agent dqn \
    --punish_needless_jump \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death