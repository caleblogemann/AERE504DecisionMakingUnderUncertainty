import numpy as np
import gym
from gym import spaces
import scipy.io
import pdb

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor, Env

import matplotlib.pyplot as plt
import os

from armClasses import *

biased = True
computeRewardAtEnd = False
target_model_update = 1e-1
nb_steps = 240000
log_interval = nb_steps/10
rmInterval = int(nb_steps/1000)
testInterval = 50
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=nb_steps)
# policy = EpsGreedyQPolicy(eps=.1)
# policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=1., value_min=.1, value_test=.05,
                              # nb_steps=nb_steps)
test_policy = GreedyQPolicy()

arm_processor = armProcessor(biased)
nb_actions = 2

# Build model
model = Sequential()
if(biased):
    model.add(Flatten(input_shape=(1,6)))
else:
    model.add(Flatten(input_shape=(1,5)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(processor=arm_processor, model=model, nb_actions=nb_actions,
        memory=memory, policy=policy, 
        target_model_update=target_model_update, gamma=1.0)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# obArray = [1.5, 2.0, 2.5]
obArray = [1.5]
cArray = [1]
fdArray = [1]
for overbooking_cost_multiplier in obArray:
    for cancellation in cArray:
        for fd in fdArray:
            outputDir = "output" + "OB" + repr(overbooking_cost_multiplier) + "c" + repr(cancellation) + "fd" + repr(fd)
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)
            trainingData = "training_cancellations" + repr(cancellation) + "_fd" + repr(fd) + ".mat"

            env = armEnv(trainingData, biased, computeRewardAtEnd, overbooking_cost_multiplier)

            info_logger = infoLogger()
            history = dqn.fit(env, callbacks=[info_logger], nb_steps=nb_steps, log_interval=log_interval)

            f = open(outputDir+"/data.txt", "w")
            f.write("Average Reward:" + repr(np.mean(info_logger.rewardPercentage[-testInterval:-1])) + "\n")
            f.write("Average Acceptance:" + repr(np.mean(info_logger.acceptPercentage[-testInterval:-1])) + "\n")
            f.write("Average Fullnes:" + repr(np.mean(info_logger.percentageFull[-testInterval:-1])) + "\n")
            f.close()

            rP = np.array(info_logger.rewardPercentage)*100
            nEpisodes = len(rP)
            t1 = np.arange(nEpisodes)
            t2 = np.arange(rmInterval/2-1, nEpisodes-rmInterval/2)
            aP = np.array(info_logger.acceptPercentage)*100
            pF = np.array(info_logger.percentageFull)*100
            np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, pF=pF)

            # plot training
            # pdb.set_trace()
            plt.plot(t1, rP, t2, running_mean(rP, rmInterval), 'r')
            plt.title('Learning Curve')
            plt.ylabel('Percentage of Optimal Reward')
            plt.xlabel('Episode')
            plt.savefig(outputDir + '/train_reward.png')
            plt.close()
            # plt.show()

            plt.plot(t1, aP, t2, running_mean(aP, rmInterval), 'r')
            plt.title('Percentage of Arrivals Accepted')
            plt.ylabel('%')
            plt.xlabel('Episode')
            plt.savefig(outputDir + '/train_arrivals.png')
            plt.close()
            # plt.show()

            plt.plot(t1, pF, t2, running_mean(pF, rmInterval), 'r')
            plt.title('Percentage of Plane that is Filled')
            plt.ylabel('%')
            plt.xlabel('Episode')
            plt.savefig(outputDir + '/train_fullness.png')
            plt.close()

            plt.plot(info_logger.seats)
            plt.title('Seat Allocation')
            plt.ylabel('Number of Seats per Class')
            plt.xlabel('Episode')
            plt.savefig(outputDir + '/seat_allocation.png')
            plt.show()

            # After training is done, we save the final weights.
            dqn.save_weights('dqn_weights.h5f', overwrite=True)

            plot_model(model, to_file='model.png', show_shapes=True)

