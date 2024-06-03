#!/bin/bash

# dqn with punish_needless_jump +stars+level+ladder
python -m main --agent dqn \
	--total_steps 20000 \
	--training_starts 1000 \
	--eval_every 2000 \
	--punish_needless_jump \
	--level_incentive \
	--ladder_incentive \
	--stars_incentive
