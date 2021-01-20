import argparse
import torch
import gym
from agents import DQNAgent, FixedDQNAgent, DoubleDQNAgent, DuelingDQNAgent
from utils import EnvWrapper, set_seed
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=['dqn', 'fixed_dqn', 'double_dqn', 'dueling_dqn'],
                        default='dqn',
                        type=str,
                        help='Model to be used. One between dqn, double_dqn and dueling_dqn (default: dqn)')

    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument('-t',
                           '--train',
                           action='store_true',
                           help='if present, train the chosen agent; if not, test it')
    arg_group.add_argument('-f',
                           '--file',
                           type=str,
                           default=None,
                           help='use the weights stored in the given file. Required if in testing mode')

    parser.add_argument('-v',
                        '--verbose',
                        choices=[0, 1, 2, 3],
                        default=2,
                        type=int,
                        help="Verbose mode. One between 0 (no plots, no logs, no video), "
                             "1 (yes plots, no logs, no video), 2 (yes plots, yes logs, no video), "
                             "3 (yes to all). Considered only in training mode (default: 2)")

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=64,
                        help='size of the batch used to perform training (default: 64)')

    parser.add_argument('-m',
                        '--memory-size',
                        type=int,
                        default=int(1e5),
                        help='maximum size of the replay memory buffer (default: 100000)')

    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discount rate for the q-values update (default: 0.99)')

    parser.add_argument('--lr',
                        type=float,
                        default=.001,
                        help='learning rate (default: 0.001)')

    parser.add_argument('--episodes',
                        type=int,
                        default=2000,
                        help='Number of episodes to perform training on.'
                             ' Considered only if in training mode (default: 2000)')

    parser.add_argument('--target-sync-freq',
                        type=int,
                        default=500,
                        help='Number of updates before the network is clones for Q-targets (default:500)')

    parser.add_argument('--learn-freq',
                        type=int,
                        default=4,
                        help='After how many steps the agent should update the weights (default: 4)')
    parser.add_argument('--decay',
                        type=float,
                        default=0.99,
                        help='Espilon decay rule for power decay (default: 0.99)')

    args = parser.parse_args()
    hyper_params = {
        'model': args.model,
        'max_timesteps': 1000,
        'target_sync_freq': args.target_sync_freq,
        'learn_freq': args.learn_freq,
        'gamma': args.gamma,
        'lr': args.lr,
        'num_episodes': args.episodes,
        'batch_size': args.batch_size,
        'memory_size': args.memory_size,
        'verbose': args.verbose,
        'max_eps': 1.0,
        'min_eps': 0.01,
        'decay_rate': args.decay,
        # 'decay_rate': 0.998,
        'decay_type': 'power_law',
        'weights_file': args.file
    }

    print("Chosen parameters:\n")
    for key in hyper_params:
        print("{}: {}".format(key, hyper_params[key]))
    print("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solved_dir = './solved/'
    plots_dir = './plots/'

    if not os.path.exists(solved_dir):
        os.mkdir(solved_dir)

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    env = gym.make("LunarLander-v2")
    seed = 18
    set_seed(seed, env)

    env_wrapper = EnvWrapper(env, device)

    if args.model == 'dqn':
        agent = DQNAgent(env_wrapper.state_dim[0],
                         env_wrapper.action_dim,
                         hyper_params['lr'],
                         hyper_params['gamma'],
                         hyper_params['memory_size'],
                         hyper_params['batch_size'],
                         hyper_params['max_eps'],
                         hyper_params['min_eps'],
                         hyper_params['decay_rate'],
                         device,
                         decay_type=hyper_params['decay_type'])

    elif args.model == "double_dqn":
        agent = DoubleDQNAgent(env_wrapper.state_dim[0],
                               env_wrapper.action_dim,
                               hyper_params['lr'],
                               hyper_params['gamma'],
                               hyper_params['memory_size'],
                               hyper_params['batch_size'],
                               hyper_params['max_eps'],
                               hyper_params['min_eps'],
                               hyper_params['decay_rate'],
                               device,
                               hyper_params['target_sync_freq'],
                               decay_type=hyper_params['decay_type'])

    else :      #  dueling_dqn
        agent = DuelingDQNAgent(env_wrapper.state_dim[0],
                                env_wrapper.action_dim,
                                hyper_params['lr'],
                                hyper_params['gamma'],
                                hyper_params['memory_size'],
                                hyper_params['batch_size'],
                                hyper_params['max_eps'],
                                hyper_params['min_eps'],
                                hyper_params['decay_rate'],
                                device,
                                hyper_params['target_sync_freq'],
                                decay_type=hyper_params['decay_type'])

    if args.train:
        paths = {
            'solved_dir': solved_dir,
            'plot_dir': plots_dir
        }

        agent.train(env_wrapper,
                    paths,
                    num_episodes=hyper_params['num_episodes'],
                    max_t=hyper_params['max_timesteps'],
                    learn_every=hyper_params['learn_freq'],
                    verbose=hyper_params['verbose'],
                    avg_period=100,
                    winning_score=200)
    else:
        paths = {
            'weights': hyper_params['weights_file'],
            'plot_dir': plots_dir
        }

        agent.test(env_wrapper,
                   paths,
                   render=True,
                   num_episodes=100,
                   max_t=1000,
                   winning_score=200)


if __name__ == "__main__":
    main()