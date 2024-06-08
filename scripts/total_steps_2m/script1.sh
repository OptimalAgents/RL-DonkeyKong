#!/bin/bash
sleep $((10 * $1))

# dqn with punish_death
python main.py --agent dqn \
    --total_steps 2000000 \
    --training_starts 80000 \
    --punish_death