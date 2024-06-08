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

If you want to play the game, run `play.py`. Start the game with `space` and move Mario using `W`, `S`, `A` and `D` keys.

## Evaluation

# Directory guide

## Structure

```bash
├── reward_plots
    ├── eval_rewards
    ├── plots
    ├── reward_plots.ipynb
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
├── eval.py
├── main.py
├── play.py
├── presentation.pptx
├── requirements.txt
```


## Description of directories

* reward_plots - directory with data and code for generating plots of reward over time including the plots.
* src - directory with all the code needed to train the agent.
* videos_for_presentation - directory with all the videos of the trained agent "playing" the game used in `presentation.pptx`.

## Notes

1. Można dać super dużą karę za bezsensowne skakanie
2. Dać mu nagrode za przejście w górę po drabinie. Zrobić to tak, że jak jest na drabinie i porusza się do góry to dostaje nagrodę, a jak porusza się w dół to dostaje karę.
   - Ze względu na to, że mamy gamme to będzie trzeba tą karę zrobić na tyle dużą, żeby warto było iść w górę. Kara > nagorda co do wartości bezwzględnej
3. Może warto zwiększyć szanse w losowaniu akcji na klknięcie w góre
   - może tylko wtedy gdy już epsilon spadnie do dolnej granicy?
   - może jakas heurystyka na to na który poziomie jesteśmy i w górą stronę chcemy iść?
4. Ewaluowac go po n-episodów, ale wtedy epsilon musi byc 0. Nie może być niedeterministyczny, chcemy mieć pewność, że to co się dzieje to wynik tego co model nauczył się do tej pory.
5. Dodać trackowanie eksperymentów przez TensorBoard
6. Może ustalić mu duża nagrodę za checkpointy, żeby szybciej się uczył?


## Notes JG

- Agents cannot take random actions. Exploration is done by epsilon exploitation outside the agent code.
- State is "rgb_array" and "human" is exaclty the same, but the "human" mode allows for displaying the progress
- Negative reward = punishment


