#!/bin/bash

# dqn with punish_needless_jump
python -m main --agent dqn \
    --punish_needless_jump \
    --total_steps 20000 \
    --training_starts 1000 \
    --eval_every 1000 \
    --punish_death