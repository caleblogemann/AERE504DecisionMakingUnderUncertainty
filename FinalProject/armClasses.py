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

    def __init__(self, filename, biased, computeRewardAtEnd, ob):
        self.biased = biased
        self.computeRewardAtEnd = computeRewardAtEnd
        self.overbooking_cost_multiplier = ob
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
        self.nArrivals = mat['nArrivals']
        self.currentDataSetIndex = None
        self.currentDataSet = None
        self.currentNArrivals = None
        self.max_reward_list = mat['maxReward']
        self.max_reward = 0

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
            if (not self.computeRewardAtEnd):
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
            if (not self.computeRewardAtEnd):
                reward -= self.fareClassPrices[classCancelled]
            # remove first element
            self.cancellations.pop(0)

        if (self.done):
            # give reward all at end
            if self.computeRewardAtEnd:
                reward = np.dot(self.seats, self.fareClassPrices)
            # compute overbooking cost
            self.overbooking = 0
            if(sum(self.seats) > self.capacity):
                number_to_bump = sum(self.seats) - self.capacity
                self.overbooking = number_to_bump
                # first bump high class
                if(number_to_bump <= self.seats[0]):
                    self.seats[0] -= number_to_bump
                    reward -= self.overbooking_cost_multiplier*self.fareClassPrices[0]*number_to_bump
                elif(number_to_bump > self.seats[0]):
                    # first high class
                    reward -= self.overbooking_cost_multiplier*self.fareClassPrices[0]*self.seats[0]
                    number_to_bump -= self.seats[0]
                    self.seats[0] = 0
                    # second middle class
                    reward -= self.overbooking_cost_multiplier*self.fareClassPrices[1]*number_to_bump
                    self.seats[1] -= number_to_bump

        self.reward = reward
        if(self.biased):
            self.observation = (self.time, self.nextClass, self.seats, 1)
        else:
            self.observation = (self.time, self.nextClass, self.seats)
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
        self.max_reward = self.max_reward_list[self.currentDataSetIndex]
        self.currentNArrivals = self.nArrivals[self.currentDataSetIndex]

        # reset variables
        self.timeIndex = 0
        self.nTimeIndices = self.currentDataSet.shape[0]
        self.time = self.currentDataSet[0,0]
        # when reading in data subtract 1 so that zero indexed
        self.nextClass = int(self.currentDataSet[0, 1] - 1)
        self.seats = np.zeros(self.nFareClasses, dtype=int)

        self.cancellations = []

        self.done = False

        if(self.biased):
            self.observation = (self.time, self.nextClass, self.seats, 1)
        else:
            self.observation = (self.time, self.nextClass, self.seats)

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
        if(self.done):
            print('Max Reward: ' + repr(self.max_reward))
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
    def __init__(self, biased):
        self.biased = biased

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """

        if(self.biased):
            processed_observation = np.concatenate(([observation[0], observation[1], observation[3]], observation[2]))
        else:
            processed_observation = np.concatenate(([observation[0], observation[1]], observation[2]))
        return processed_observation 

class infoLogger(Callback):
    def _set_env(self, env):
        self.env = env

    def on_train_begin(self, logs={}):
        self.seats = []
        self.episode_reward = []
        self.max_reward = []
        self.rewardPercentage = []
        self.overbooking = []
        self.nArrivals = []
        self.acceptPercentage = []
        self.percentageFull = []

    def on_episode_begin(self, episode, logs={}):
        self.nAccepts = 0

    def on_step_end(self, epsode_steps, logs={}):
        if(logs['action'] == 1):
            self.nAccepts += 1

    def on_episode_end(self, episode, logs={}):
        self.seats.append(self.env.seats)
        self.episode_reward.append(logs['episode_reward'])
        self.max_reward.append(self.env.max_reward)
        self.overbooking.append(self.env.overbooking)
        totalArrivals = sum(self.env.currentNArrivals)
        self.nArrivals.append(totalArrivals)
        # pdb.set_trace()
        self.rewardPercentage.append(logs['episode_reward']/self.env.max_reward)
        self.acceptPercentage.append(self.nAccepts/totalArrivals)
        if(self.env.overbooking > 0):
            self.percentageFull.append(1 + self.env.overbooking/self.env.capacity)
        else:
            maxPossiblePassengers = min(totalArrivals, self.env.capacity)
            self.percentageFull.append(sum(self.env.seats)/maxPossiblePassengers)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N
