import minerl
import copy
from logging import getLogger
from collections import OrderedDict, deque
import os

import gym
import numpy as np
import cv2

from pfrl.wrappers import ContinuingTimeLimit, RandomizeAction, Monitor
from pfrl.wrappers.atari_wrappers import ScaledFloatFrame, LazyFrames

cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)

datasetA = minerl.data.make("MineRLObtainDiamondVectorObf-v0", num_workers=1)
datasetB = minerl.data.make("MineRLObtainDiamond-v0", num_workers=1)
traj_name="v3_villainous_black_eyed_peas_loch_ness_monster-2_3997-48203"
itrA = datasetA.load_data(traj_name)
itrB = datasetB.load_data(traj_name)
# itr = dataset.batch_iter(1, 1, -1)

class MockEnv(gym.Env):
    def __init__(self, kmeans=None):
        self.mock_obf = OrderedDict([
            ('pov', np.random.randint(256, size=(64, 64, 3), dtype=np.uint8)),
            ('vector', np.random.rand(64).astype(np.float32)),
        ])
        self.observation_space = gym.spaces.Dict(
            OrderedDict([
                ('pov', gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)),
                ('vector', gym.spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)),
            ])
        )
        self.action_space = gym.spaces.Dict(
            OrderedDict([
                ('vector', gym.spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)),
            ])
        )
        self.count = 0
        self.kmeans = kmeans

    def reset(self):
        self.count = 0
        return self.mock_obf

    def step(self, action):
        """
        action = {'vector': np.random.rand(64) * 2 - 1}
        for i in range(20000):
            action = next(itrA)[1]
            print("MineRLObtainDiamondVectorObf")
            print(minerl.herobraine.envs.MINERL_OBTAIN_DIAMOND_OBF_V0.unwrap_action(action))
            if self.kmeans is not None:
                nearest_idx = self.kmeans.predict([action['vector']])[0]
                recov = self.kmeans.cluster_centers_[nearest_idx]
                print("MineRLObtainDiamondVectorObf (nearest k-means)")
                print(minerl.herobraine.envs.MINERL_OBTAIN_DIAMOND_OBF_V0.unwrap_action({'vector': recov}))
            action_b = next(itrB)[1]
            print("MineRLObtainDiamond")
            print(action_b)
            while not (action_b['craft'] == 'none'):
                pass
            print("_____")
        assert False
        """
        self.count += 1
        return self.mock_obf, np.random.rand(), self.count > 8000, {}
