#Perry A., Joshua D., Neal P.

import gym
import os
import numpy as np
import matplotlib.pyplot as plt

num_state_vars = 8
num_action_vars = 4

num_trials_to_retrain = 2
nn_epochs = 4
num_steps_per_trial = 5000
num_landing_trials = 1000
max_global_memory = 1000000

num_random_trials = 20
num_init_trials = 10
to_load_weights = True

all_total_rewards = []

env = gym.make('LunarLander-v2')

def attempt_landing(env, seed=None, visualize=False):

    # Iterate through games
    for trial in range(num_landing_trials):
        total_reward = 0
        curr_state = env.reset()
        for step in range(num_steps_per_trial):
            # Get an action
            a = env.action_space.sample()
            if (trial % 20 == 0):
               env.render()

            # Take the action
            next_state, reward, complete, _ = env.step(a)  # updates current state to next state
            total_reward += reward

            if complete:
                all_total_rewards.append(total_reward)

                if (trial % 20 == 0):
                    print('Trial ', trial)

                break

if __name__ == '__main__':

    attempt_landing(env, visualize=True)

    plt.figure()
    plt.plot(range(num_landing_trials), all_total_rewards)
    average = np.mean(all_total_rewards)
    print(average)
    plt.title("Total Reward vs Trial Number for Random Action")
    plt.xlabel("Trial Number")
    plt.ylabel("Total Reward")
    plt.savefig("random_results.png")
