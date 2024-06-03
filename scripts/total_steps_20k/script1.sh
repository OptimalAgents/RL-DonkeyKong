#!/bin/bash

# dqn with punish_death
python -m main --agent dqn \
	--total_steps 20000 \
	--training_starts 1000 \
	--eval_every 2000 \
	--punish_death
