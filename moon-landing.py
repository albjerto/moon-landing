import torch
import gym
from agent import Agent
from utils import EnvWrapper, set_seed
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2")

seed = 18

set_seed(seed, env)

# num_frames = 75000
memory_size = int(1e5)
batch_size = 64
target_update = 100
# decay_rate = 1 / 50000      # for linear
decay_rate = 0.99         # for power law
# decay_rate = 0.001            # for exponential
max_eps = 1.
min_eps = .01
gamma = .99
lr = .001
plot_freq = 200
winning_score = 200
avg_period = 150
episodes = 1000
max_t = 1000
test_episodes = 100


if not os.path.exists('./solved'):
    os.mkdir('./solved')

if not os.path.exists('./plots'):
    os.mkdir('./plots')


agent = Agent(env.observation_space.shape[0],
              env.action_space.n,
              lr,
              gamma,
              memory_size,
              batch_size,
              max_eps,
              min_eps,
              decay_rate,
              device,
              decay_type="power_law")

epsilons = []
losses = []
scores = []
score = 0
updates = 0
env_wrapper = EnvWrapper(env, device)

for ep in range(1, episodes + 1):
    state = env_wrapper.reset()
    score = 0
    for t in range(1, max_t + 1):
        # env_wrapper.render()
        action = agent.choose_action(state)
        next_state, reward, done = env_wrapper.step(action)
        agent.remember(state, action, reward, next_state, done)
        score += reward
        state = next_state

        if agent.can_sample_from_mem():
            loss = agent.update()
            losses.append(loss)
            updates += 1

            if updates % target_update == 0:
                agent.target_hard_update()

        if done:
            agent.update_eps()
            break

    scores.append(score)
    epsilons.append(agent.curr_eps)
    print("\n****************************\n"
          "Episode {} finished. Score: {}. Current exploration rate: {}\n"
          "****************************\n".format(ep, score, agent.curr_eps))
    if ep % plot_freq == 0:
        agent.plot(scores,
                   avg_period,
                   winning_score,
                   losses,
                   epsilons,
                   filename="./plots/DQN_checkpoint_" + str((ep / plot_freq) + 1) + '.png')

    avg_reward = np.mean(scores[-avg_period:])
    if avg_reward >= winning_score:
        print("\n\n\nEnvironment solved after {} episodes, with an avg reward of {}. Highest reward was {}.\n"
              .format(ep, avg_reward, np.max(scores)))
        agent.policy_net.save('./solved/DQN_policy_net.pth')
        break

agent.plot(scores, avg_period, winning_score, losses, epsilons, filename='./plots/DQN_final.png')

# qs_history = np.array([np.array(x)
# for x in agent.q_table_history if agent.q_table_history.index(x) not in (5, 7, 11, 12)], dtype=object)
#
# for qs in qs_history:
#     _, ax = plt.subplots(nrows=1, ncols=4)
#     for i in range(len(ax)):
#         ax[i].plot(qs[:, i])
#     plt.show()


scores = []
print("TESTING")
for episode in range(1, test_episodes + 1):
    done = False
    state = env_wrapper.reset()
    score = 0
    t = 0
    while not done and t < max_t:
        env_wrapper.render()
        action = agent.choose_action(state, testing=True)
        new_state, reward, done = env_wrapper.step(action)
        score += reward
        state = new_state
        t += 1
    scores.append(score)

    print("Testing episode {} finished\n".format(episode))

print(scores)
