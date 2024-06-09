# Human level performance of an AI playing Donkey Kong

This repository contains a project that aims to train a reinforcement learning model to play the video game Donkey Kong using Deep Q-Networks (DQN). In this game the player takes on the role of Mario and tries to rescue his girlfriend from a giant gorilla called Donkey Kong. In order to get to his girlfriend, Mario needs to climb the unbroken ladders and omit the barrels. In addition, if Mario grabs the hammer, he can hit the barrels. 

<p align="center">
  <img src="https://github.com/OptimalAgents/RL-DonkeyKong/blob/main/donkey_kong.png" alt="Donkey Kong screenshot"/>
</p>

We made this project for a Case Study course at Warsaw University of Technology. Here is the repository for this course:
[https://github.com/PrzeChoj/2024Lato-WarsztatyBadawcze](https://github.com/PrzeChoj/2024Lato-WarsztatyBadawcze)

# Team members
* Jakub Grzywaczewski ([@ZetrextJG](https://github.com/ZetrextJG))
* Anna Ostrowska ([@annaostrowska03](https://github.com/annaostrowska03))
* Igor Rudolf ([@IgorRudolf](https://github.com/IgorRudolf))
* Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))

# Function execution guide

This guide provides instructions on how to execute the functions included in this repository.

## Training agent

If you want to train the agent, just run the `main.py` script. You can choose the algorithm between `q-learning` and `sarsa`. We recommend using the default option which is `q-learning`. 

## Playing the game

If you want to play the game, run `play.py`. Start the game with `space` and move Mario using `W`, `S`, `A` and `D` keys. After starting the game, use `space` for jumping.

# Directory guide

## Structure

```bash
├── reward_plots
    ├── eval_rewards
    ├── plots
    ├── reward_plots.ipynb
├── runs
├── scripts
    ├── total_stes_20k
    ├── total_steps_2m
    ├── exp1.sh
    ├── exp2.sh
    ├── methods_to_check
├── src
    ├── methods
    │   ├── __init__.py
    │   ├── dqn
    ├── Agents.py
    ├── __init__.py
    ├── env.py
    ├── utils.py
├── videos_for_presentation
├── .gitignore
├── LICENSE
├── README.md
├── donkey_kong.png
├── main.py
├── play.py
├── presentation.pptx
├── report.pdf
├── requirements.txt
```


## Description of directories

* reward_plots - directory with data and code for generating plots of reward over time including the plots.
* src - directory with all the code needed to train the agent.
* runs - results of running the scripts in directory `scripts`.
* scripts - scripts launching `main.py` with appropriate parameters.
* videos_for_presentation - directory with all the videos of the trained agent "playing" the game used in `presentation.pptx`.
