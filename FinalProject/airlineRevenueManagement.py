import numpy as np
import gym
from gym import spaces
import scipy.io
import pdb

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.core import Env

import matplotlib.pyplot as plt


class armEnv(Env):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    To implement your own environment, you need to define the following methods:
    - `step`
    - `reset`
    - `render`
    - `close`
    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """

    def __init__(self, filename):
        mat = scipy.io.loadmat(filename, squeeze_me=True)

        self.reward_range = (-np.inf, np.inf)

        self.action_space = spaces.Discrete(2)

        self.nFareClasses = mat['nFareClasses']
        self.capacity = mat['capacity']
        self.classSizeMean = mat['classSizeMean']
        self.classCancelRate = mat['classCancelRate']
        self.totalTime = mat['totalTime']
        self.fareClassPrices = mat['fareClassPrices']

        self.nDataSets = mat['nDataSets']
        self.dataSets = mat['dataSets']
        self.currentDataSetIndex = None
        self.currentDataSet = None

        self.timeIndex = 0
        self.nTimeIndices = 0
        self.time = 0
        # when reading in data subtract 1 so that zero indexed
        self.nextClass = 0 
        self.seats = np.zeros(self.nFareClasses, dtype=int)

        self.cancellations = []

        self.done = False

        # observation (time, nextClass, seats)
        self.observation_space = spaces.Tuple((spaces.Box(low=0,high=self.totalTime, shape=(1,), dtype=float), spaces.Discrete(self.nFareClasses), spaces.Box(low=np.zeros(self.nFareClasses), high=np.full(self.nFareClasses, 2*self.capacity), dtype=int)))

        # TODO: add storage for list of arrival times

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        if(self.done):
            return None

        reward = 0

        self.action = action

        # if accepted add to seats
        if(action == 1):
            # pdb.set_trace()
            self.seats[self.nextClass] += 1
            reward += self.fareClassPrices[self.nextClass]
            # check if passenger will cancel
            cancellationTime = self.currentDataSet[self.timeIndex, 2]
            if (cancellationTime > 0):
                self.cancellations.append((cancellationTime, self.nextClass))
                # sort on first index cancellation time
                self.cancellations.sort(key= lambda elem: elem[0])

        # set new time and nextClass
        if(self.timeIndex < self.nTimeIndices - 1):
            self.timeIndex += 1
            self.time = self.currentDataSet[self.timeIndex, 0]
            self.nextClass = int(self.currentDataSet[self.timeIndex, 1] - 1)
        else:
            self.done = True
            self.time = self.totalTime
            self.nextClass = -1;

        # remove cancellations
        while(len(self.cancellations) > 0 and self.cancellations[0][0] < self.time):
            classCancelled = self.cancellations[0][1]
            self.seats[classCancelled] -= 1
            reward -= self.fareClassPrices[classCancelled]
            # remove first element
            self.cancellations.pop(0)

        if (self.done):
            # compute overbooking cost
            overbooking_cost_multiplier = 2
            if(sum(self.seats) > self.capacity):
                number_to_bump = sum(self.seats) - self.capacity
                # first bump high class
                if(number_to_bump <= self.seats[0]):
                    self.seats[0] -= number_to_bump
                    reward -= overbooking_cost_multiplier*self.fareClassPrices[0]*number_to_bump
                elif(number_to_bump > self.seats[0]):
                    # first high class
                    reward -= overbooking_cost_multiplier*self.fareClassPrices[0]*self.seats[0]
                    number_to_bump -= self.seats[0]
                    self.seats[0] = 0
                    # second middle class
                    reward -= overbooking_cost_multiplier*self.fareClassPrices[1]*number_to_bump
                    self.seats[1] -= number_to_bump

        self.reward = reward
        self.observation = (self.time, self.nextClass, self.seats, 1.0)
        return self.observation, reward, self.done, dict()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        # load next Data Set
        if (self.currentDataSetIndex == None):
            self.currentDataSetIndex = 0
        else:
            self.currentDataSetIndex += 1

        if(self.currentDataSetIndex >= self.nDataSets):
            print('No More Data Sets')

        self.currentDataSet = self.dataSets[self.currentDataSetIndex]

        # reset variables
        self.timeIndex = 0
        self.nTimeIndices = self.currentDataSet.shape[0]
        self.time = self.currentDataSet[0,0]
        # when reading in data subtract 1 so that zero indexed
        self.nextClass = int(self.currentDataSet[0, 1] - 1)
        self.seats = np.zeros(self.nFareClasses, dtype=int)

        self.cancellations = []

        self.done = False

        self.observation = (self.time, self.nextClass, self.seats, 1.0)
        return self.observation


    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        # outfile.write('State: ' + repr(self.observation) + ' Action: ' + repr(self.action) + '\n')
        if(self.action == 0):
            print('Denied!!')
        return print('State: ' + repr(self.observation) + ' Action: ' + repr(self.action) + ' Reward: ' + repr(self.reward))

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

class armProcessor(Processor):
    """Abstract base class for implementing processors.
    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """

        return np.concatenate(([observation[0], observation[1], observation[3]], observation[2]))


nFareClasses = 3
arm_processor = armProcessor()

env = armEnv("test.mat")
nb_actions = env.action_space.n

# Build model
model = Sequential()
model.add(Flatten(input_shape=(1,3+nFareClasses)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(processor=arm_processor, model=model, nb_actions=nb_actions, memory=memory,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = dqn.fit(env, nb_steps=80000, log_interval=8000)

# After training is done, we save the final weights.
dqn.save_weights('dqn_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5)
plt.plot(history.history['episode_reward'])
plt.title('Episode Reward During Training')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()
