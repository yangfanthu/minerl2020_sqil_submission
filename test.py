import json
import select
import time
import logging
import os
import threading
from mod.env_wrappers import wrap_env
from mod.cached_kmeans import cached_kmeans
from reward_boundary_calculator import cached_reward_boundary
from mod.sqil import get_agent
from mod.data.pipeline_wrapper import DataPipelineWrapper


from typing import Callable

import aicrowd_helper
import gym
import minerl
import abc
import numpy as np

import coloredlogs
coloredlogs.install(logging.DEBUG)

# Agent settings
GPU = -1

ARCH = 'dueling'
KMEANS_N_CLUSTERS = 30
KMEANS_N_CLUSTERS_VC = 60
KMEANS_SEED = 0
OPTION_N_GROUPS = 10
FINAL_EPSILON = 0.01
FINAL_EXPLORATION_FRAMES = 10 ** 6
LR = 0.0000625
ADAM_EPS = 0.00015
PRIORITIZED = True
UPDATE_INTERVAL = 4
REPLAY_CAPACITY = 300000
NUM_STEP_RETURN = 1
GAMMA = 0.99
REPLAY_START_SIZE = 5000
TARGET_UPDATE_INTERVAL = 10000
CLIP_DELTA = True
BATCH_ACCUMULATOR = 'mean'
EXP_REWARD_SCALE = 10
EXPERIENCE_LAMBDA = 1
FRAME_SKIP = 4
GRAY_SCALE = False
FRAME_STACK = 4
EVAL_EPSILON = 0.001

maximum_frames = 8000000
STEPS = maximum_frames // FRAME_SKIP

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))

class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.

    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class

    This class enables the evaluator to run your agent in parallel,
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################

class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent.
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64*64*3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size))*2 -1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten()/255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1,1)}


    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs,reward,done,_ = single_episode_env.step(self.act(self.flatten_obs(obs)))


class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)

class MineRLSQILBaselineAgent(MineRLAgentBase):
    def __init__(self, env):
        self.env = env

    def load_agent(self):
        boundaries = cached_reward_boundary(
            cache_dir='./train/boundary_cache/',
            env_id=MINERL_GYM_ENV,
            n_groups=OPTION_N_GROUPS,
            random_state=KMEANS_SEED)
        reward_channel_scale = 1. / boundaries[-1]
        self.agent = get_agent(
            n_actions=self.env.action_space.n, arch=ARCH, n_input_channels=self.env.observation_space.shape[0],
            final_epsilon=FINAL_EPSILON,
            final_exploration_frames=FINAL_EXPLORATION_FRAMES, explorer_sample_func=self.env.action_space.sample,
            lr=LR, adam_eps=ADAM_EPS,
            steps=STEPS, update_interval=UPDATE_INTERVAL,
            replay_capacity=REPLAY_CAPACITY, num_step_return=NUM_STEP_RETURN,
            gpu=GPU, gamma=GAMMA, replay_start_size=REPLAY_START_SIZE,
            target_update_interval=TARGET_UPDATE_INTERVAL, clip_delta=CLIP_DELTA,
            batch_accumulator=BATCH_ACCUMULATOR, expert_dataset=None,
            exp_reward_scale=EXP_REWARD_SCALE, experience_lambda=EXPERIENCE_LAMBDA,
            reward_boundaries=boundaries, reward_channel_scale=reward_channel_scale,
        )

        self.agent.load(os.path.abspath(os.path.join(__file__, os.pardir, 'train', 'latest', 'best')))

    def run_agent_on_episode(self, single_episode_env: Episode):
        with self.agent.eval_mode():
            obs = single_episode_env.reset()
            while True:
                a = self.agent.act(obs)
                obs, r, done, info = single_episode_env.step(a)

#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   #
######################################################################
AGENT_TO_TEST = MineRLSQILBaselineAgent  # MineRLMatrixAgent # MineRLMatrixAgent, MineRLRandomAgent, YourAgentHere
# AGENT_TO_TEST = MineRLRandomAgent


####################
# EVALUATION CODE  #
####################
def main():
    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    # Create the parallel envs (sequentially to prevent issues!)
    kmeans_normal = cached_kmeans(
        cache_dir='./train/kmeans_cache/',
        env_id=MINERL_GYM_ENV,
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=KMEANS_SEED,
        sample_by_trajectory=True,
        only_vector_converter=False)
    kmeans_vector_converter = cached_kmeans(
        cache_dir='./train/kmeans_cache/',
        env_id=MINERL_GYM_ENV,
        n_clusters=KMEANS_N_CLUSTERS_VC,
        random_state=KMEANS_SEED,
        sample_by_trajectory=True,
        only_vector_converter=True)

    def wrapper(env):
        return wrap_env(
            env=env, test=True, monitor=False, outdir=None,
            env_id=MINERL_GYM_ENV,  # added
            frame_skip=FRAME_SKIP, gray_scale=GRAY_SCALE, frame_stack=FRAME_STACK,
            randomize_action=False, eval_epsilon=EVAL_EPSILON,
            action_choices=kmeans_normal.cluster_centers_,
            action_choices_vector_converter=kmeans_vector_converter.cluster_centers_,
            append_reward_channel=(OPTION_N_GROUPS > 1),
        )

    envs = [wrapper(gym.make(MINERL_GYM_ENV)) for _ in range(EVALUATION_THREAD_COUNT)]
    # envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    agent = AGENT_TO_TEST(envs[0])
    # agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent()

    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # A simple funciton to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(episodes_per_thread[i]):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass

    evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    for thread in evaluator_threads:
        thread.start()

    # wait fo the evaluation to finish
    for thread in evaluator_threads:
        thread.join()

if __name__ == "__main__":
    main()
