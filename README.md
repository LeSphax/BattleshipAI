# BattleshipAI

This repository contains a gym environment with a very simplified version of the game Battleship.

It also contains a script to train an AI to play that game with QLearning.

## Install

Clone the repository `git clone git@github.com:LeSphax/BattleshipAI.git`

Install the dependencies `pip3 install -e BattleshipAI`

Get inside the directory `cd BattleshipAI`

## Usage
The way the script is setup is that it will first train the algorithm and log how many turns a game lasts in average.
After the training it will show the algorithm playing until you press ctrl+c.

`python3 main.py random` to run a random player, it should finish a game in 61 turns on average.

`python3 main.py ai` to train the ai then see it playing. 

It should train for a few minutes until getting to less than 20 turns on average.
You can see that once it finds a ship it will correctly search around and finish the game in the next 2-5 turns.

## Game

The game is a 2 player 5x5 version of battleship with a single 3x1 ship that is positioned randomly.

On each players turn he will see the 5x5 board of his opponent with the ship being hidden.

He can also see the moves he has already played on this board and their effects (MISSED/TOUCHED).

The player then have to choose one position on the board that he wants to shoot at.

The game ends when a player has shot the 3 cells where his opponent's ship is located.

#### Improvements
Increase the size of the board and the number of ships.
Allow the AI to set the position of its ships instead of placing them randomly.

## AI

This is a very simplified version of QLearning. The AI gets the board of his opponent as input and 
outputs a matrix of the same size with a prediction on each cell of the reward it should get next turn by shooting the cell.

To pick an action we use an epsilon greedy algorithm.

So by default the cell with the highest predicted value will be selected but there is also a small chance to pick a random cell instead.

#### Limitations

Since the algorithm is only predicting the reward for the next turn, it can't create longer term strategies.
By looking longer term, an algorithm could have a more effective strategy to search the position of the ship.

At the moment it simply learns not to hit cells that it has already hit before and to search the other parts of a ship once it has touched it.

Another problem is that the algorithm is completely deterministic except for the epsilon-greedyness. 
Making it easy to exploit for his opponent in a real self-play scenario where ais could position their ships.
