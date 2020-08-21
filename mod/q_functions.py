import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pfrl import action_value
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction
from pfrl.q_functions.dueling_dqn import constant_bias_initializer
from pfrl.initializers import init_chainer_default


def parse_arch(arch, n_actions, n_input_channels, reward_boundaries=None,
               reward_channel_scale=1.):
    if arch == 'dueling':
        # Conv2Ds of (channel, kernel, stride): [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        return DuelingDQN(n_actions, n_input_channels=n_input_channels)
    elif arch == 'dueling_option':
        assert reward_boundaries is not None
        return OptionDuelingDQN(n_actions, n_input_channels=n_input_channels,
                                reward_boundaries=reward_boundaries,
                                reward_channel_scale=reward_channel_scale)
    elif arch == 'dueling_med':
        return DuelingDQNMed(n_actions, n_input_channels=n_input_channels)
    elif arch == 'distributed_dueling':
        n_atoms = 51
        v_min = -10
        v_max = 10
        return DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_input_channels)
    else:
        raise RuntimeError('Unsupported architecture name: {}'.format(arch))


class DuelingDQN(nn.Module, StateQFunction):
    """Dueling Q-Network
    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # Modified from 3136 -> 1024
        self.a_stream = MLP(1024, n_actions, [512])
        self.v_stream = MLP(1024, 1, [512])

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)


class OptionDuelingDQN(nn.Module, StateQFunction):
    """Dueling Q-Network with option based on cumulative rewards
    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1,
                 reward_boundaries=None, reward_channel_scale=1.):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.boundaries = torch.from_numpy(np.array(reward_boundaries)) * reward_channel_scale - 1e-8

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # Modified from 3136 -> 1024
        self.a_streams = nn.ModuleList([MLP(1024, n_actions, [512]) for _ in range(len(self.boundaries) + 1)])
        self.v_streams = nn.ModuleList([MLP(1024, 1, [512]) for _ in range(len(self.boundaries) + 1)])

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        rewards = x[:, -1, 0, 0]
        policy_indices = torch.as_tensor([torch.sum(reward > self.boundaries.to(rewards.device)) for reward in rewards])

        # Advantage
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = []
        for index, policy_index in enumerate(policy_indices):
            ya.append(self.a_streams[policy_index](h[index:(index + 1)]))
        ya = torch.cat(ya, axis=0)

        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = []
        for index, policy_index in enumerate(policy_indices):
            ys.append(self.v_streams[policy_index](h[index:(index + 1)]))
        ys = torch.cat(ys, axis=0)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)


class DuelingDQNMed(nn.Module, StateQFunction):
    """Dueling Q-Network
    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # Modified from 3136 -> 1024
        self.med_stream = MLP(1024, 256, [512])
        self.a_stream = MLP(320, n_actions, [256])
        self.v_stream = MLP(320, 1, [256])

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        batch_size = x.shape[0]
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))
        h = h.reshape(batch_size, -1)
        h = self.med_stream(h)
        h = torch.cat([h, x[:, -1, 0, :]], 1)

        # Advantage
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)


class DistributionalDuelingDQN(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_input_channels=4,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # ここだけ変える必要があった
        # self.main_stream = nn.Linear(3136, 1024)
        self.main_stream = nn.Linear(1024, 1024)
        self.a_stream = nn.Linear(512, n_actions * n_atoms)
        self.v_stream = nn.Linear(512, n_atoms)

        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))

        mean = ya.sum(dim=1, keepdim=True) / self.n_actions

        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(x.device)
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)
