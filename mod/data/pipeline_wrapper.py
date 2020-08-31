from logging import getLogger
import numpy as np
from collections import deque, OrderedDict
from pfrl.wrappers.atari_wrappers import LazyFrames
from data.action_converter import DualKMeansActionConverter


logger = getLogger(__name__)


def _trim_first_dim_in_dict(input):
    ret = {}
    for key in input.keys():
        ret[key] = input[key][0]
    return ret


def _get_skipped_obs(obs, frameskip, done_index=None):
    assert len(obs) % frameskip == 0
    ret = []
    for i in range(0, len(obs), frameskip):
        if done_index is not None and i <= done_index:
            ret.append(obs[-frameskip])
        else:
            ret.append(obs[i])
    return np.array(ret)


def _append_reward_channel(obs, cumulative_rewards, scale):
    channel_shape = obs.shape[-2:]
    reward_channels = np.ones((obs.shape[0], 1, *channel_shape), dtype=np.float32) * scale * cumulative_rewards.reshape(-1, 1, 1, 1)  # noqa
    return np.concatenate([obs, reward_channels], axis=1)


def _get_aggregated_action(obs, actions, next_obs, frameskip):
    # Return an aggregated action of the last frameskip frames
    # If frames contains actions converting observation vector,
    #     it returns the first one of them.
    # Otherwise it returns the first frame action.
    assert len(obs['vector']) >= frameskip
    assert len(actions) >= frameskip
    assert len(next_obs['vector']) >= frameskip
    is_normal = np.all(
        np.isclose(obs['vector'][-frameskip:],
                   next_obs['vector'][-frameskip:]),
        axis=1)
    return actions[-frameskip:][np.argmin(is_normal)]


class DataPipelineWrapper:
    def __init__(self, dataset, observation_converters,
                 action_converters, frameskip=1, framestack=1,
                 append_reward_channel=False, reward_scale=1./1024):
        self.dataset = dataset
        self.observation_converters = observation_converters
        self.action_converters = action_converters
        self.frameskip = frameskip
        self.framestack = framestack
        self.append_reward_channel = append_reward_channel
        self.episode_names = self.dataset.get_trajectory_names()
        self.current_episode_name = np.random.choice(self.episode_names)
        self.batch_loader = self.dataset.load_data(self.current_episode_name)
        self.current_reward_sum = 0
        self.reward_scale = reward_scale

    def _get_next(self):
        obs_pov, obs_vector, actions, rewards, next_pov, next_vector, dones, cumulative_rewards, cumulative_next_rewards \
            = [[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]]

        dn = False
        while len(rewards[0]) < self.frameskip * self.framestack:
            ob, ac, rw, nob, dn = next(self.batch_loader)
            self.current_reward_sum += rw

            obs_pov[0].append(ob['pov'])
            obs_vector[0].append(ob['vector'])
            actions[0].append(ac['vector'])
            rewards[0].append(rw)
            next_pov[0].append(nob['pov'])
            next_vector[0].append(nob['vector'])
            dones[0].append(dn)
            cumulative_rewards[0].append(self.current_reward_sum - rw)
            cumulative_next_rewards[0].append(self.current_reward_sum)

            if dn:
                # reset episode
                self.current_episode_name = np.random.choice(self.episode_names)
                self.batch_loader = self.dataset.load_data(self.current_episode_name)
                self.current_reward_sum = 0

        return (
            OrderedDict([('pov', np.array(obs_pov)), ('vector', np.array(obs_vector))]),
            OrderedDict([('vector', np.array(actions))]),
            np.array(rewards, dtype=np.float32),
            OrderedDict([('pov', np.array(next_pov)), ('vector', np.array(next_vector))]),
            np.array(dones),
            np.array(cumulative_rewards, dtype=np.float32),
            np.array(cumulative_next_rewards, dtype=np.float32))

    def sample(self):
        obs, actions, rewards, next_obs, dones, cumulative_rewards, cumulative_next_rewards = self._get_next()
        logger.debug(actions)
        # Convert
        trim_obs = _trim_first_dim_in_dict(obs)
        trim_next_obs = _trim_first_dim_in_dict(next_obs)
        c_obs = trim_obs
        c_next_obs = trim_next_obs
        for converter in self.observation_converters:
            c_obs = converter(c_obs)
            c_next_obs = converter(c_next_obs)
        if self.append_reward_channel:
            c_obs = _append_reward_channel(c_obs, cumulative_rewards, self.reward_scale)
            c_next_obs = _append_reward_channel(c_next_obs, cumulative_next_rewards, self.reward_scale)
        c_actions = _trim_first_dim_in_dict(actions)
        for converter in self.action_converters:
            if isinstance(converter, DualKMeansActionConverter):
                c_actions = converter(trim_obs, c_actions, trim_next_obs)
            else:
                c_actions = converter(c_actions)
        rewards = rewards[0]
        dones = dones[0]
        # Find done index
        if len(dones) > self.frameskip and np.any(dones[:-self.frameskip]):
            done_index = np.argmax(
                dones[:-self.frameskip]
                * np.arange(len(dones[:-self.frameskip])))
            aggregated_rewards = np.sum(rewards[(done_index + 1):])
        else:
            done_index = None
            aggregated_rewards = np.sum(rewards)
        # Aggregation for Framestack, Frameskip
        skipped_obs = _get_skipped_obs(c_obs, self.frameskip, done_index)
        skipped_next_obs = np.concatenate((skipped_obs[1:], c_next_obs[-1:]),
                                          axis=0)
        aggregated_action = _get_aggregated_action(
            trim_obs, c_actions, trim_next_obs, self.frameskip)
        if self.framestack == 1:
            return (
                skipped_obs[0], aggregated_action, aggregated_rewards,
                c_next_obs[-1], np.any(dones[-self.frameskip:]))
        else:
            return (
                LazyFrames(skipped_obs, 0), aggregated_action,
                aggregated_rewards, LazyFrames(skipped_next_obs, 0),
                np.any(dones[-self.frameskip:]))
