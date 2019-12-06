#Perry A., Joshua D., Neal P.

import gym
import keras
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras import optimizers

num_state_vars = 8
num_action_vars = 4
learning_rate = 1e-3
model_weights_filename = 'LL_NN_Weights.h5'

discount_factor = 0.98
start_epsilon_prob = 0.05
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

class Memory:
  def __init__(self, state, q_action):
    self.state = state
    self.q_action = q_action

global_memory = Memory(np.zeros((1,num_state_vars)), np.zeros((1,num_action_vars)))

model = Sequential()

model.add(Dense(512, activation='relu', input_dim = num_state_vars))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_action_vars, activation='linear'))

opt = optimizers.adam(lr = learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

def load_weights():
    dir_path = os.path.realpath(".")
    fn = dir_path + "/" + model_weights_filename
    if  os.path.isfile(fn):
        model.load_weights(model_weights_filename)

def get_q(state):
    s = np.zeros((1, num_state_vars))
    s[0] = state
    predicted_q = np.array(model.predict(s)[0])
    return predicted_q.reshape((1,num_action_vars))

def attempt_landing(env, seed=None, visualize=False):

    # Iterate through games
    for trial in range(num_landing_trials):
        total_reward = 0
        trial_memory = Memory(np.zeros((1,num_state_vars)), np.zeros((1,num_action_vars)))
        curr_state = env.reset()
        for step in range(num_steps_per_trial):
            # Get an action
            if trial < num_random_trials+num_init_trials:
                a = env.action_space.sample()
            else:
                # Epsilon greedy
                epsilon_prob = start_epsilon_prob - (start_epsilon_prob/num_landing_trials)*trial
                if np.random.random(1) < epsilon_prob:
                    a = env.action_space.sample()
                else:
                    # Q-approximation
                    q_values = get_q(curr_state)
                    a = np.argmax(q_values)
            if (trial % 20 == 0):
               env.render()

            # Take the action
            next_state, reward, complete, _ = env.step(a)  # updates current state to next state
            total_reward += reward

            # Calculate the Q-Value

            if trial < num_init_trials:
                q_values = np.zeros((1,num_action_vars))
                q_values[0,a] = reward
            elif trial < (num_random_trials+num_init_trials):
                q_values = get_q(curr_state)
                print(q_values.shape)
                previous_q = q_values[0,a]
                print(previous_q)
                print(np.max(get_q(next_state)))
                q_values[0,a] = previous_q + learning_rate*(reward + discount_factor*np.max(get_q(next_state))-previous_q)
            else:
                previous_q = q_values[0,a]
                q_values[0,a] = previous_q + learning_rate*(reward + discount_factor*np.max(get_q(next_state))-previous_q)

            q_values = q_values.reshape((1,4))

            # Update trial memory
            if step == 0:
                trial_memory.state = curr_state
                trial_memory.q_action = q_values
            else:
                trial_memory.state = np.vstack((trial_memory.state, curr_state))
                trial_memory.q_action = np.vstack((trial_memory.q_action, q_values))

            if complete:
                all_total_rewards.append(total_reward)

                # Update global memory
                if (trial == 0):
                    global_memory.state = trial_memory.state
                    global_memory.q_action = trial_memory.q_action
                else:
                    global_memory.state = np.vstack((global_memory.state, trial_memory.state))
                    global_memory.q_action = np.vstack((global_memory.q_action, trial_memory.q_action))

                # If we exceed memory capacity, delete initial memory entries
                if np.alen(global_memory.state) > max_global_memory:
                    global_memory.state = global_memory.state[np.alen(trial_memory.state):]
                    global_memory.q_action = global_memory.q_action[np.alen(trial_memory.q_action):]

                if trial >= num_init_trials-1:
                    if (trial % 20 == 0):
                        print('Trial ', trial)
                    model.fit(global_memory.state, global_memory.q_action, batch_size=1024, nb_epoch=nn_epochs, verbose=0)
                    model.save_weights(model_weights_filename)
                    load_weights()

                break

if __name__ == '__main__':

    if to_load_weights:
        load_weights()

    attempt_landing(env, visualize=True)

    plt.figure()
    plt.plot(range(num_landing_trials), all_total_rewards)
    average = np.mean(all_total_rewards)
    print(average)
    plt.title("Total Reward vs Trial Number for Deep Q-Learned Action")
    plt.xlabel("Trial Number")
    plt.ylabel("Total Reward")
    plt.savefig("deep_q_results.png")
