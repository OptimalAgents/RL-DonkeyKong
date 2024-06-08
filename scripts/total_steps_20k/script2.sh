#!/bin/bash
sleep $((10 * $1))

# dqn with level_incentive
python -m main --agent dqn \
	--level_incentive \
	--total_steps 20000 \
	--training_starts 1000 \
	--eval_every 1000 \
	--punish_death

