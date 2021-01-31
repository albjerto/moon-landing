import gym
import torch


class EnvWrapper:
    """
    Environment wrapper that handles numpy to torch conversions.
    """
    def __init__(self, env, device):
        self.env = None
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env

        self.state = None
        self.device = device
        self.unwrapped = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

    def process_state(self, state):
        """
        Convert state to torch tensor.
        :param state: state to be converted
        :return: converted state
        """
        return torch.tensor(state, dtype=torch.float, device=self.device)

    def step(self, action):
        """
        Perform an action.
        :param action: action chosen
        :return:
            - next_state: processed new state of the environment
            - reward: reward for the action performed
            - done: True if next_state is terminal
        """
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.process_state(next_state)
        self.state = next_state
        return next_state, reward, done

    def reset(self):
        """
        Resets the environment to the starting state.
        :return: the processed state
        """
        self.state = self.process_state(self.env.reset())
        return self.state

    def render(self):
        """ Render the environment """
        self.env.render()

    def close(self):
        """ Close the environment """
        self.env.close()
