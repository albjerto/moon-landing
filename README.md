# Landing on the moon with Deep Q-Learning :rocket::moon: #
This repository contains the code of the project for the module Autonomous and Adaptive Systems,
held by Prof. [Mirco Musolesi](https://www.mircomusolesi.org/) at the University of Bologna.

The purpose of this project is to compare, both in terms of training stability and testing performances, one of the most well-known algorithms for Deep Reinforcement Learning, Deep Q-Learning, and its improvements. In particular, the experiments will train the base DQN, the DQN with fixed Q-targets, the Double DQN and the Dueling Double DQN, and test them on the Lunar Lander environment from OpenAI gym. The results of the experiments show that, while the extensions to the base algorithm improved its performances greatly, the simpleness of the environment prevents us from identifying a clear winner.

Read the paper-like report for the project [here](https://github.com/albjerto/moon-landing/blob/master/minipaper.pdf).

![agent perfomance after training](./img/test_performance.gif "Agent performance after training")

## Dependencies ##
- [PyTorch](https://pytorch.org/)
- [OpenAI gym](https://gym.openai.com/)
- [NumPy](https://numpy.org/) 
- [Matplotlib](https://matplotlib.org/)
- [Box2d](https://box2d.org/)

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
python3 -m venv ./venv
source ./venv/Scripts/activate
pip3 install -r requirements.txt 
```

## Project structure ##
- [./solved/](#usage)
- [./moon-landing.py](#moon-landingpy)
- [./agents.py](#agents)
- [./models.py](#models)
- [./utils.py](#utilspy)

### Moon-landing.py ###
This file contains the main function, which creates the agents and launches them.

### Agents ###
The agents are implemented in the `agents.py` file. The available agents are:
- DQNAgent
- FixedDQNAgent
- DoubleDQNAgent
- DuelingDQNAgent

All the agents implement a `BaseAgent`, which contains the basic functions common to all of them. Since many of the improvements are also incremental, `FixedDQNAgent` extends `DQNAgent`, and `DoubleDQNAgent` extends `FixedDQNAgent`.

### Models ###
The available models are implemented in the file `models.py`:
- DQN
- Dueling DQN

### utils.py ###
This file contains utility functions and classes:
- Replay Memory, used to store experiences in the form `e = (state, action, reward, new_state, done)`
- Environment Wrapper, which handles the conversions between numpy arrays and torch tensors

## Usage ##
Running
```
python3 moon-landing.py -h
```
displays a help message with all the customizable parameters and their syntax. 
```
usage: moon-landing.py [-h] [--model {dqn,fixed_dqn,double_dqn,dueling_dqn}] (-t | -f FILE) [-v {0,1,2,3}] [-r] [-b BATCH_SIZE] [-m MEMORY_SIZE] [--gamma GAMMA] [--lr LR] [--episodes EPISODES]
                       [--target-sync-freq TARGET_SYNC_FREQ] [--learn-freq LEARN_FREQ] [--decay DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --model {dqn,fixed_dqn,double_dqn,dueling_dqn}
                        Model to be used. One between dqn, double_dqn and dueling_dqn (default: dqn)
  -t, --train           if present, train the chosen agent; if not, test it
  -f FILE, --file FILE  Use the weights stored in the given file. Required if in testing mode
  -v {0,1,2,3}, --verbose {0,1,2,3}
                        Verbose mode. One between 0 (no plots, no logs, no video), 1 (yes plots, no logs, no video), 2 (yes plots, yes logs, no video), 3 (yes to all). Considered only in training mode
                        (default: 2)
  -r, --render          Render video of the environment. Considered only in test mode
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        size of the batch used to perform training (default: 64)
  -m MEMORY_SIZE, --memory-size MEMORY_SIZE
                        maximum size of the replay memory buffer (default: 100000)
  --gamma GAMMA         discount rate for the q-values update (default: 0.99)
  --lr LR               learning rate (default: 0.001)
  --episodes EPISODES   Number of episodes to perform training on. Considered only if in training mode (default: 2000)
  --target-sync-freq TARGET_SYNC_FREQ
                        Number of updates before the network for Q-targets is clones (default: 500)
  --learn-freq LEARN_FREQ
                        After how many steps the agent should update the weights (default: 4)
  --decay DECAY         Espilon decay rule for power decay (default: 0.99)
```

For example, to train a DQN agent, run
```
python3 moon-landing.py -t --model dqn
```
If the agent solves the environment, the weights that achieved the highest 100-episodes mean will be saved as a `.pth` file in the `./solved` folder, with the name of the model and a timestamp with the format `YYYYmmdd_HHMM`.
Then, to test it, run
```
python3 moon-landing.py --model dqn -f ./path/to/file
```
In the folder `./solved` there are the weight files of some pre-trained agents.
