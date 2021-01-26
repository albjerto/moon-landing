import gym
import torch


class EnvWrapper:
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
        return torch.tensor(state, dtype=torch.float, device=self.device)

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.process_state(next_state)
        self.state = next_state
        return next_state, reward, done

    def reset(self):
        self.state = self.process_state(self.env.reset())
        return self.state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
