# Landing on the moon with Deep Reinforcement Learning #
This repository contains the code of the project for the module Autonomous and Adaptive Systems,
held by Prof. [Mirco Musolesi](https://www.mircomusolesi.org/) at the University of Bologna.

## Dependencies ##

- Python 3.6+
- PyTorch
- OpenAI gym
- NumPy 
- Matplotlib
- Box2d

## Installation ##
To install this project, run 
```
git clone https://github.com/albjerto/moon-landing
cd moon-landing
pip3 install -r requirements.txt
```

Alternatively, it is also possible to install the dependencies in a virtual environment:
```
git clone https://github.com/albjerto/moon-landing
cd moon-landing
python3 -m venv .venv
./Scripts/activate
pip3 install -r requirements.txt 
```

## Project structure ##
- [moon-landing.py](#Moon-landing-py)
- [agents.py](#agents)
- [models.py](#models)
- [utils.py](#utils)

### Moon-landing.py ###

## Usage ##
Running
```
python3 moon-landing.py -h
```
displays a help message with all the customizable parameters and their syntax. 
```
usage: moon-landing.py [-h] [--model {dqn,double_dqn,dueling_dqn}] (-t | -f FILE) [-v {0,1,2,3}] [-b BATCH_SIZE] [-m MEMORY_SIZE] [--gamma GAMMA] [--lr LR] [--episodes EPISODES]

optional arguments:
  -h, --help            show this help message and exit
  --model {dqn,double_dqn,dueling_dqn}
                        Model to be used, between dqn, double_dqn and dueling_dqn
  -t, --train           if present, train the chosen agent; if not, test it
  -f FILE, --file FILE  use the weights stored in the given file. Required if in testing mode
  -v {0,1,2,3}, --verbose {0,1,2,3}
                        Verbose mode. One between 0 (no plots, no logs, no video), 1 (yes plots, no logs, no video), 2 (yes plots, yes logs, no video), 3 (yes to all). Considered only in training mode
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        size of the batch used to perform training (default: 64)
  -m MEMORY_SIZE, --memory-size MEMORY_SIZE
                        maximum size of the replay memory buffer (default: 100000)
  --gamma GAMMA         discount rate for the q-values update (default: 0.99)
  --lr LR               learning rate (default: 0.001)
  --episodes EPISODES   Number of episodes to perform training on. Considered only if in training mode.
```

For example, to train a DQN agent, run
```
python3 moon-landing.py -t --model dqn
```
If the agent solves the environment, its weights will be saved as a `.pth` file in the `./solved` folder.
Then, to test it, run
```
python3 moon-landing --model dqn -f ./path/to/file
```
In the folder `./solved` there are the weight files of some pre-trained agents.
