#!/bin/bash

# dqn with punish_needless_jump +stars
python -m main --agent dqn \
	--punish_needless_jump \
	--stars_incentive \
	--total_steps 20000 \
	--training_starts 1000 \
	--eval_every 1000 \
	--punish_death

