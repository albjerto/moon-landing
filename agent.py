from models.dqn import DQN
import torch.optim as optim
import torch.nn.functional as F
import math
from utils import *
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 lr,
                 gamma,
                 max_memory_size,
                 batch_size,
                 eps_start,
                 eps_end,
                 eps_decay,
                 device,
                 decay_type="linear"):

        def linear(curr_eps):
            return max(
                eps_end, curr_eps - ((
                        eps_start - eps_end
                ) * eps_decay)
            )

        def power_law_decay(curr_eps):
            return max(
                eps_end, curr_eps * eps_decay
            )

        def exponential_decay(curr_step):
            return max(
                eps_end, eps_end + (eps_start - eps_end) * math.exp(-1. * curr_step * eps_decay)
            )

        decay_options = {
            "linear": linear,
            "power_law": power_law_decay,
            "exp": exponential_decay
        }

        if decay_type not in decay_options:
            raise ValueError("decay_type parameter not valid. Expected one of {}"
                             .format([i for i in decay_options.keys()]))

        self.decay_law = decay_options[decay_type]
        self.decay_type = decay_type

        self.device = device

        # networks
        self.output_dim = output_dim
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        # policy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.curr_step = 0
        self.curr_eps = eps_start

        # replay memory
        self.memory = ReplayMemory(max_memory_size)
        self.batch_size = batch_size

    def update_eps(self):
        if self.decay_type == 'exp':
            self.curr_eps = self.decay_law(self.curr_step)
        else:
            self.curr_eps = self.decay_law(self.curr_eps)

        return self.curr_eps

    def remember(self, s, a, r, s_, d):
        self.memory.push(Experience(s, a, r, s_, d))

    def target_hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def target_soft_update(self, tau=0.001):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def choose_action(self, state, testing=False):
        self.curr_step += 1

        if not testing and np.random.random() < self.curr_eps:
            return np.random.randint(0, self.output_dim)
        else:
            # we're using the network for inference only, we don't want to track the gradients in this case
            with torch.no_grad():
                return self.policy_net(state).argmax().detach().item()

    def can_sample_from_mem(self):
        return self.memory.can_sample(self.batch_size)

    def update(self):
        sample = self.memory.sample(self.batch_size)

        batch = Experience(*zip(*sample))
        states = torch.stack(batch.state).to(self.device)
        next_states = torch.stack(batch.new_state).to(self.device)
        actions = torch.LongTensor(batch.action).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).reshape(-1, 1).to(self.device)
        dones = torch.FloatTensor(batch.done).reshape(-1, 1).to(self.device)

        curr_q_vals = self.policy_net(states).gather(1, actions)
        next_q_vals = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        target = (rewards + self.gamma * next_q_vals * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(curr_q_vals, target)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()
        # for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        return loss.item()

    @staticmethod
    def plot(scores, avg_period, winning_score, losses, eps=None, filename=None):
        # plt.figure(figsize=(15, 5))
        # plt.subplot(131)
        # plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        # plt.plot(scores)
        # plt.subplot(132)
        # plt.title('loss')
        # plt.plot(losses)
        # plt.subplot(133)
        # plt.title('epsilons')
        # plt.plot(epsilons)
        # plt.show()

        def moving_avg():
            avg = []
            for i in range(len(scores)):
                if i < avg_period:
                    avg.append(0)
                else:
                    avg.append(np.mean(scores[i - avg_period:i]))

            return avg

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(18, 5))
        axis1.clear()
        axis1.plot(scores, 'blue', label='Score per episode', alpha=0.4)
        axis1.plot(moving_avg(), 'black',
                   label='Mean score of the last {} episodes'.format(avg_period))
        axis1.axhline(winning_score, c='green', label='Winning score', alpha=0.7)
        axis1.axhline(0, c='grey', ls='--', alpha=0.7)
        axis1.set_xlabel('Episodes')
        axis1.set_ylabel('Scores')
        axis1.legend()

        if eps is not None:
            tw_axis1 = axis1.twinx()
            tw_axis1.plot(eps, 'red', alpha=0.5)
            tw_axis1.set_ylabel('Epsilon', color='red')

        axis2.plot(losses)
        axis2.set_title('Loss')

        if filename is not None:
            plt.savefig(filename)

        plt.show()

        plt.close()
