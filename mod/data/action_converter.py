"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""
import minerl
import copy
import numpy as np
import gym
from collections import OrderedDict
from sklearn.cluster import KMeans
from logging import getLogger

logger = getLogger(__name__)


class VectorActionConverter:
    def __call__(self, action):
        assert 'vector' in action
        return action['vector']

    def convert_space(self, space):
        return space['vector']

    def invert(self, action):
        return OrderedDict([
            ('vector', action)
        ])


class VectorDiscretizeConverter:
    def __init__(self, num_discretize=7, low=-1, high=1):
        self.num_discretize = 7
        self.low = low
        self.high = high

    def __call__(self, action):
        assert 'vector' in action
        scale = self.high - self.low
        action = np.round(
            (action['vector'] - self.low) / scale * (self.num_discretize - 1))
        return action.astype(np.int32)

    def convert_space(self, space):
        return gym.spaces.Box(
            low=np.zeros_like(space['vector'].low),
            high=(np.ones_like(space['vector'].low) * self.num_discretize - 1),
            dtype=np.int32)

    def invert(self, action):
        scale = self.high - self.low
        inverted = action / (self.num_discretize - 1) * scale + self.low
        return OrderedDict([
            ('vector', inverted)
        ])


class KMeansActionConverter:
    def __init__(self, kmeans):
        self.kmeans = kmeans
        keys = ['craft', 'nearbyCraft', 'nearbySmelt', 'equip', 'place', 'attack', 'forward', 'camera']
        for center in kmeans.cluster_centers_:
            action = {'vector': center}
            dict = minerl.herobraine.envs.MINERL_OBTAIN_DIAMOND_OBF_V0.unwrap_action(action)

    def __call__(self, action):
        index = self.kmeans.predict(action['vector'])
        return index

    def convert_space(self, space):
        return gym.spaces.Discrete(self.kmeans.n_clusters)

    def invert(self, action):
        return OrderedDict([
            ('vector', self.kmeans.cluster_centers_[action])
        ])


class DualKMeansActionConverter:
    def __init__(self, kmeans_normal, kmeans_vector_converter):
        self.kmeans_normal = kmeans_normal
        self.kmeans_vector_converter = kmeans_vector_converter

    def __call__(self, obs, action, next_obs):
        is_normal = np.all(np.isclose(obs['vector'], next_obs['vector']), axis=1)
        idx_normal = self.kmeans_normal.predict(action['vector'])
        idx_vc = self.kmeans_vector_converter.predict(action['vector'])
        num_action_normal = self.kmeans_normal.n_clusters
        index = idx_normal * is_normal + (idx_vc + num_action_normal) * np.logical_not(is_normal)  # noqa
        return index

    def convert_space(self, space):
        return gym.spaces.Discrete(self.kmeans.n_clusters)

    def invert(self, action):
        return OrderedDict([
            ('vector', self.kmeans.cluster_centers_[action])
        ])
