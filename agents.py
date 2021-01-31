from abc import abstractmethod

import torch.optim as optim
import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.common import Experience
from utils.replay import ReplayMemory
from models import DQN, DuelingDQN


class BaseAgent:
    """
    Basic agent abstract class.
    """
    def __init__(self,
                 max_memory_size,
                 batch_size,
                 eps_start,
                 eps_end,
                 eps_decay,
                 device,
                 decay_type="linear"):

        self.model_name = "Base"

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

        # policy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.curr_step = 0
        self.curr_eps = eps_start

        # replay memory
        self.memory = ReplayMemory(max_memory_size, device)
        self.batch_size = batch_size

    def update_eps(self):
        if self.decay_type == 'exp':
            self.curr_eps = self.decay_law(self.curr_step)
        else:
            self.curr_eps = self.decay_law(self.curr_eps)

        return self.curr_eps

    def remember(self, s, a, r, s_, d):
        """
        Store in replay memory as an Experience tuple.
        :param s: starting state
        :param a: action taken while in state s
        :param r: reward received for action a
        :param s_: new state of the environment after a
        :param d: tells if s_ is terminal or not
        """
        self.memory.append(Experience(s, a, r, s_, d))

    def can_sample_from_mem(self):
        """
        Check if the Replay Memory has enough samples for a batch.
        """
        return self.memory.can_sample(self.batch_size)

    @abstractmethod
    def choose_action(self, state, testing=False):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def set_train(self):
        pass

    @abstractmethod
    def set_test(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @staticmethod
    def plot(scores, avg_period, winning_score, eps=None, filename=None):
        """
        Plot the current trend of the agent.
        :param scores: list of total rewards per episode to plot
        :param avg_period: period of episodes over which to calculate moving avg
        :param winning_score: threshold for winning one episode
        :param eps: if specified, plots also the epsilon-value
        :param filename: if specified, saves the plot at the path given
        """
        def moving_avg():
            avg = []
            for i in range(len(scores)):
                if i < avg_period:
                    avg.append(0)
                else:
                    avg.append(np.mean(scores[i - avg_period:i]))

            return avg

        fig, axis = plt.subplots()
        axis.clear()
        axis.plot(scores, 'blue', label='Score per episode', alpha=0.4)
        axis.plot(moving_avg(), 'black',
                  label='Mean score of the last {} episodes'.format(avg_period))
        axis.axhline(winning_score, c='green', label='Winning score', alpha=0.7)
        axis.axhline(0, c='grey', ls='--', alpha=0.7)
        axis.set_xlabel('Episodes')
        axis.set_ylabel('Scores')
        axis.legend()

        if eps is not None:
            tw_axis = axis.twinx()
            tw_axis.plot(eps, 'red', alpha=0.5)
            tw_axis.set_ylabel('Epsilon', color='red')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close()

    def train(self,
              env,
              paths,
              num_episodes=2000,
              max_t=1000,
              learn_every=4,
              verbose=2,
              avg_period=100,
              winning_score=200,
              plot_freq=200):
        """
        Trains the agent. If the environment is solved, the weights
        corresponding to the highest mean will be saved.

        :param env: environment chosen
        :param paths: dictionary that contains:
            - solved_dir: directory in which to save winning weight files
            - plots_dir: directory in which to save plots
        :param num_episodes: number of episodes to perform training
        :param max_t: max timesteps per episode
        :param learn_every: how often should the networks weight be updated
        :param verbose: number between 0 and 3, controls how much log info
            is wanted
        :param avg_period: period over which compute the avg of the rewards
        :param winning_score: threshold for winning one episode
        :param plot_freq: how often should plots be printed
        """

        scores = []
        losses = []
        epsilons = []
        updates = 0
        last_mean = 0
        avg_rewards = []

        self.set_train()
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        for ep in range(1, num_episodes + 1):
            state = env.reset()
            score = 0
            for t in range(1, max_t + 1):
                if verbose == 3:
                    env.render()

                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                score += reward
                state = next_state

                if self.can_sample_from_mem() and self.curr_step % learn_every == 0:
                    loss = self.learn()
                    losses.append(loss)
                    updates += 1

                if done:
                    self.update_eps()
                    break

            scores.append(score)
            epsilons.append(self.curr_eps)
            if verbose > 1:
                print("\n****************************\n"
                      "Episode {} finished. Score: {}. Current exploration rate: {}\n"
                      "****************************\n".format(ep, score, self.curr_eps))

            if ep % plot_freq == 0 and verbose > 0:
                # plotting to see behaviour: not saving to file
                self.plot(scores,
                          avg_period,
                          winning_score,
                          epsilons)

            avg_reward = np.mean(scores[-avg_period:])
            avg_rewards.append(avg_reward)
            if avg_reward > winning_score and avg_reward > last_mean:
                last_mean = avg_reward
                if verbose > 1:
                    print("\n\nNew best mean: {} at episode {}.\n"
                          .format(avg_reward, ep))

                # self.save(paths['solved_dir'] + self.model_name + '_solved_' + curr_time + '.pth')
                self.save(paths['solved_dir']
                          + self.model_name
                          + '_'
                          + curr_time
                          + '_best.pth')

                # final plot: saving to file
                self.plot(scores,
                          avg_period,
                          winning_score,
                          epsilons,
                          filename=paths['plot_dir'] + self.model_name + '_train_' + curr_time + '.png')

        if verbose > 0:
            print("Training finished.")
            self.plot(scores,
                      avg_period,
                      winning_score,
                      epsilons)
        env.close()

    def test(self,
             env,
             paths,
             render=True,
             num_episodes=100,
             max_t=1000,
             winning_score=200):
        """
        Test the agent with the given weights.
        :param env: environment chosen
        :param paths: dictionary that contains:
            - weights: path to the weight file
            - plots: directory in which to save plots
        :param render: if True, show the agent playing
        :param num_episodes: number of episodes to test the agent on
        :param max_t: max timesteps per episode
        :param winning_score: threshold score to win an episode
        """
        try:
            self.load(paths['weights'])
        except FileNotFoundError:
            print("File not found.")
            exit(1)

        self.set_test()

        test_scores = []
        print("TESTING")
        for episode in range(1, num_episodes + 1):
            s = env.reset()
            score = 0
            for t in range(1, max_t + 1):
                if render:
                    env.render()

                a = self.choose_action(s, testing=True)
                s_, r, d = env.step(a)
                score += r
                s = s_
                t += 1
                if d:
                    break

            test_scores.append(score)

            print("Episode {} - score {}\n".format(episode, score))

        if paths['plot_dir'] is not None:
            plt.axhline(winning_score, c='green', label='Winning score', alpha=0.7)
            plt.plot(test_scores, c='blue', label='Score per episode')
            curr_time = datetime.now().strftime("%Y%m%d_%H%M")
            plt.savefig(paths['plot_dir'] + self.model_name + '_test_' + curr_time + '.png')
        print("Testing finished.")
        test_scores = np.array(test_scores)
        success = test_scores[test_scores >= 200]
        print("Success rate: {}% - highest score: {}"
              .format(len(success) * num_episodes / 100, np.max(test_scores)))

        env.close()


class DQNAgent(BaseAgent):
    """
    Agent with a DQN network.
    """
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
                 linear1_units=64,
                 linear2_units=64,
                 decay_type="linear"):

        super().__init__(max_memory_size,
                         batch_size,
                         eps_start,
                         eps_end,
                         eps_decay,
                         device,
                         decay_type)

        self.model_name = "DQN"
        self.output_dim = output_dim
        self.policy_net = DQN(input_dim, output_dim, linear1_units, linear2_units).to(device)

        # optimizer
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def choose_action(self, state, testing=False):
        """
        Choose an action to perform. Uses eps-greedy approach.
        :param state: current state of the environment
        :param testing: if True, always choose greedy action
        :return: the action chosen
        """
        self.curr_step += 1

        if not testing and np.random.random() < self.curr_eps:
            return np.random.randint(0, self.output_dim)
        else:
            # we're using the network for inference only, we don't want to track the gradients in this case
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def learn(self):
        """
        Update the weights of the network.
        :return: the loss
        """
        states, next_states, actions, rewards, dones = self.memory.sample(self.batch_size)

        curr_q_vals = self.policy_net(states).gather(1, actions)
        next_q_vals = self.policy_net(next_states).max(1, keepdim=True)[0].detach()
        target = (rewards + self.gamma * next_q_vals * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(curr_q_vals, target)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()

        return loss.item()

    def set_test(self):
        """ Sets the network in evaluation mode """
        self.policy_net.eval()

    def set_train(self):
        """ Sets the network in training mode """
        self.policy_net.train()

    def save(self, filename):
        """
        Save the network weights.
        :param filename: path
        """
        self.policy_net.save(filename)

    def load(self, filename):
        """
        Load the network weights.
        :param filename: path of the weight file
        """
        self.policy_net.load(filename, self.device)


class FixedDQNAgent(DQNAgent):
    """
    DQN Agent with a target network to compute Q-targets.
    Extends DQNAgent.
    """
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
                 target_update=100,
                 linear1_units=64,
                 linear2_units=64,
                 decay_type="linear"):

        super().__init__(input_dim,
                         output_dim,
                         lr,
                         gamma,
                         max_memory_size,
                         batch_size,
                         eps_start,
                         eps_end,
                         eps_decay,
                         device,
                         linear1_units,
                         linear2_units,
                         decay_type)

        self.model_name = "FixedDQN"

        self.target_update_freq = target_update
        # networks
        self.output_dim = output_dim
        self.target_net = DQN(input_dim, output_dim, linear1_units, linear2_units).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.updated = 0

    def learn(self):
        """
        Update the weights of the network, using
        target_net to compute Q-targets. Every self.target_update_freq
        updates, clone the policy_net.
        :return: the loss
        """
        states, next_states, actions, rewards, dones = self.memory.sample(self.batch_size)

        curr_q_vals = self.policy_net(states).gather(1, actions)
        next_q_vals = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        target = (rewards + self.gamma * next_q_vals * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(curr_q_vals, target)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()

        self.updated += 1

        if self.updated % self.target_update_freq == 0:
            self.target_hard_update()

        return loss.item()

    def target_hard_update(self):
        """ Clone the policy net weights into the target net """
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DoubleDQNAgent(FixedDQNAgent):
    """
    DoubleDQN Agent. Extends FixedDQNAgent.
    """
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
                 target_update=100,
                 linear1_units=64,
                 linear2_units=64,
                 decay_type="linear"):

        super().__init__(input_dim,
                         output_dim,
                         lr,
                         gamma,
                         max_memory_size,
                         batch_size,
                         eps_start,
                         eps_end,
                         eps_decay,
                         device,
                         target_update,
                         linear1_units,
                         linear2_units,
                         decay_type)

        self.model_name = "DoubleDQN"

    def learn(self):
        """
        Update the weights of the network, using
        target_net and policy_net to compute Q-targets.
        Every self.target_update_freq updates, clone the policy_net.
        :return: the loss
        """
        states, next_states, actions, rewards, dones = self.memory.sample(self.batch_size)

        curr_q_vals = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_vals_idx = self.policy_net(next_states).argmax(1)

        next_q_vals = self.target_net(next_states).gather(1, max_next_q_vals_idx.unsqueeze(1)).detach()
        target = (rewards + self.gamma * next_q_vals * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(curr_q_vals, target)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()
        self.updated += 1

        if self.updated % self.target_update_freq == 0:
            self.target_hard_update()
        # for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        return loss.item()


class DuelingDQNAgent(BaseAgent):
    """
    DuelingDQN agent. Extends BaseAgent and uses the
    dueling architecture for both policy and target networks.
    """
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
                 target_update,
                 linear_units=64,
                 advantage_units=32,
                 value_units=32,
                 decay_type="linear"):

        super().__init__(max_memory_size,
                         batch_size,
                         eps_start,
                         eps_end,
                         eps_decay,
                         device,
                         decay_type=decay_type)

        self.output_dim = output_dim

        self.policy_net = DuelingDQN(input_dim, output_dim, linear_units, advantage_units, value_units).to(device)
        self.target_net = DuelingDQN(input_dim, output_dim, linear_units, advantage_units, value_units).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        self.updated = 0
        self.model_name = self.policy_net.name

        self.target_update_freq = target_update

    def learn(self):
        """
        Update the weights of the network, using
        target_net and policy_net to compute Q-values. Every self.target_update_freq
        updates, clone the policy_net.
        :return: the loss
        """
        states, next_states, actions, rewards, dones = self.memory.sample(self.batch_size)

        curr_q_vals = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_vals_idx = self.policy_net(next_states).argmax(1)

        next_q_vals = self.target_net(next_states).gather(1, max_next_q_vals_idx.unsqueeze(1)).detach()
        target = (rewards + self.gamma * next_q_vals * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(curr_q_vals, target)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()
        self.updated += 1

        if self.updated % self.target_update_freq == 0:
            self.target_hard_update()
        # for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        return loss.item()

    def save(self, filename):
        """
        Save policy network weights to file.
        :param filename: path
        """
        self.policy_net.save(filename)

    def load(self, filename):
        """
        Load network weights from file.
        :param filename: path of the weight file
        """
        self.policy_net.load(filename, self.device)

    def choose_action(self, state, testing=False):
        """
        Chooses an action. Uses eps-greedy approach.
        :param state: current state of the environment
        :param testing: if True, always choose greedy action
        :return: the action chosen
        """
        self.curr_step += 1
        if not testing and np.random.random() < self.curr_eps:
            return np.random.randint(0, self.output_dim)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def set_train(self):
        """ Set the network in training mode """
        self.policy_net.train()

    def set_test(self):
        """ Set the network in evaluation mode """
        self.policy_net.eval()

    def target_hard_update(self):
        """ Clone the policy net weights into the target net """
        self.target_net.load_state_dict(self.policy_net.state_dict())
