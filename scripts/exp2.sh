#!/bin/bash

# Run the experiment
python -m src/main --agent dqn \
	--ladded_incentive \
	--level_incentive \
	--heuristic_actions
