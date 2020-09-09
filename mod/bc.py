import os
import logging
import argparse

import numpy as np
import torch
import minerl  # noqa: register MineRL envs as Gym envs.
import gym

import pfrl


# local modules
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import utils
from env_wrappers import wrap_env
from q_functions import parse_arch
from cached_kmeans import cached_kmeans
from reward_boundary_calculator import cached_reward_boundary
from data.pipeline_wrapper import DataPipelineWrapper
from data.observation_converter import (
    GrayScaleConverter, PoVOnlyConverter, VectorCombineConverter,
    MoveAxisConverter, ScaledFloatConverter)
from data.action_converter import (
    VectorActionConverter, VectorDiscretizeConverter,
    KMeansActionConverter, DualKMeansActionConverter)
from agents.sqil import SQIL

from bc_module import QFunction
from tensorboardX import SummaryWriter
import datetime

import pdb

logger = logging.getLogger(__name__)
writer = SummaryWriter(logdir=('results/{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def main(argv=None):
    parser = argparse.ArgumentParser()

    env_choices = [
        # basic envs
        'MineRLTreechop-v0',
        'MineRLNavigate-v0', 'MineRLNavigateDense-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
        'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
        'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
        # obfuscated envs
        'MineRLTreechopVectorObf-v0',
        'MineRLNavigateVectorObf-v0', 'MineRLNavigateExtremeVectorObf-v0', 'MineRLNavigateDenseVectorObf-v0', 'MineRLNavigateExtremeDenseVectorObf-v0',
        'MineRLObtainDiamondVectorObf-v0', 'MineRLObtainDiamondDenseVectorObf-v0',
        'MineRLObtainIronPickaxeVectorObf-v0', 'MineRLObtainIronPickaxeDenseVectorObf-v0',
        # for debugging
        'MineRLNavigateDenseFixed-v0', 'MineRLObtainTest-v0',
    ]
    parser.add_argument('--env', type=str, choices=env_choices, required=True,
                        help='MineRL environment identifier.')
    parser.add_argument('--steps', type=int, default=8000000,
                        help='Number of training steps for training. Note that steps of evaluation episodes are not counted.')
    parser.add_argument('--eval-interval', type=int, default=600000,
                        help='Number of training steps between two performance evaluations.')

    # meta settings
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--eval-n-runs', type=int, default=3)
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    parser.add_argument('--remove-timestamp', action='store_true', default=False,
                        help='Save results not to outdir/timestamp but to outdir/latest.')

    # training scheme (agent)
    parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'DoubleDQN', 'PAL', 'CategoricalDoubleDQN'])

    # network architecture
    parser.add_argument('--arch', type=str, default='dueling', choices=['dueling', 'dueling_med', 'distributed_dueling', 'dueling_option'],
                        help='Network architecture to use.')

    # update rule settings
    parser.add_argument('--update-interval', type=int, default=4, help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--frame-skip', type=int, default=4, help='Number of frames skipped (None for disable).')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate.')
    parser.add_argument('--no-clip-delta', dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam.')
    parser.add_argument('--batch-accumulator', type=str, default='sum', choices=['sum', 'mean'], help='accumulator for batch loss.')

    # observation conversion related settings
    parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    parser.add_argument('--frame-stack', type=int, default=1, help='Number of frames stacked (None for disable).')

    # exploration related settings
    parser.add_argument('--final-exploration-frames', type=int, default=10 ** 6,
                        help='Timesteps after which we stop annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01, help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001, help='Exploration epsilon used during eval episodes.')
    # parser.add_argument('--noisy-net-sigma', type=float, default=None,
    #                     help='NoisyNet explorer switch. This disables following options: '
    #                     '--final-exploration-frames, --final-epsilon, --eval-epsilon')

    # experience replay buffer related settings
    parser.add_argument('--replay-capacity', type=int, default=10 ** 6, help='Maximum capacity for replay buffer.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before performing gradient updates.')
    # parser.add_argument('--prioritized', action='store_true', default=False, help='Use prioritized experience replay.')

    # target network related settings
    parser.add_argument('--target-update-interval', type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which the target network is updated.')

    # K-means related settings
    parser.add_argument('--kmeans-n-clusters', type=int, default=30, help='#clusters for K-means')
    parser.add_argument('--dual-kmeans', action='store_true', default=False,
                        help='Use dual kmeans of different clustering criteria.')
    parser.add_argument('--kmeans-n-clusters-vc', type=int, default=30, help='#clusters for Dual K-means of vector converters')

    # SQIL specific settings
    parser.add_argument('--max-episode-len', type=int, default=None, help='Manual maximum episode length.')
    parser.add_argument('--exp-reward-scale', type=float, default=1, help='Expert reward scale to control randomness.')
    parser.add_argument('--experience-lambda', type=float, default=1, help='Weight coefficient of batches from experiences.')
    parser.add_argument('--option-n-groups', type=int, default=1, help='Number of options to switch polices.')
    parser.add_argument('--num-actions', type=int, default = 30, help='num of acitons can be taken')
    args = parser.parse_args(args=argv)
    if args.remove_timestamp:
        args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir, exp_id='latest')
    else:
        args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)

    import logging
    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(args.outdir, 'log.txt'), format=log_format, level=args.logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    logger.info('Output files are saved in {}'.format(args.outdir))

    try:
        _main(args)
    except:  # noqa
        logger.exception('execution failed.')
        raise


def _main(args):
    os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = args.outdir

    # Set a random seed used in ChainerRL.
    pfrl.utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args.seed

    # K-Means
    if args.dual_kmeans:
        kmeans_normal = cached_kmeans(
            cache_dir=os.environ.get('KMEANS_CACHE'),
            env_id=args.env,
            n_clusters=args.kmeans_n_clusters,
            random_state=args.seed,
            sample_by_trajectory=True,
            only_vector_converter=False)
        kmeans_vector_converter = cached_kmeans(
            cache_dir=os.environ.get('KMEANS_CACHE'),
            env_id=args.env,
            n_clusters=args.kmeans_n_clusters_vc,
            random_state=args.seed,
            sample_by_trajectory=True,
            only_vector_converter=True)
    else:
        kmeans = cached_kmeans(
            cache_dir=os.environ.get('KMEANS_CACHE'),
            env_id=args.env,
            n_clusters=args.kmeans_n_clusters,
            random_state=args.seed)
    if args.option_n_groups > 1:
        boundaries = cached_reward_boundary(
            cache_dir=os.environ.get('BOUNDARY_CACHE'),
            env_id=args.env,
            n_groups=args.option_n_groups,
            random_state=args.seed)
        reward_channel_scale = 1. / boundaries[-1]
    else:
        boundaries = None
        reward_channel_scale = 1.

    # Prepare data processor
    orig_data = minerl.data.make(args.env, num_workers=1)

    observation_converters = []
    if args.gray_scale:
        observation_converters.append(GrayScaleConverter())
    if args.env.find('VectorObf') != -1:
        observation_converters.append(VectorCombineConverter())
    elif args.env.startswith('MineRLNavigate'):
        raise NotImplementedError()
    else:
        observation_converters.append(PoVOnlyConverter())
    observation_converters.append(MoveAxisConverter())
    observation_converters.append(ScaledFloatConverter())
    if args.dual_kmeans:
        action_converters = [DualKMeansActionConverter(kmeans_normal, kmeans_vector_converter)]  # noqa
    else:
        action_converters = [KMeansActionConverter(kmeans)]
    if args.demo:
        experts = None  # dummy
    else:
        experts = DataPipelineWrapper(
            orig_data,
            observation_converters=observation_converters,
            action_converters=action_converters,
            frameskip=args.frame_skip, framestack=args.frame_stack,
            append_reward_channel=(args.option_n_groups > 1),
            reward_scale=reward_channel_scale)
    # temp = experts.sample()

    q_function = QFunction(n_actions = args.num_actions).to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(q_function.parameters(), lr=learning_rate)  
    criterion = torch.nn.CrossEntropyLoss()
    max_epochs = 50
    total_steps = 2500
    batch_size = 32

    n_batch_train = 0
    for epoch in range(max_epochs):
        for step in range(total_steps):
            obs_list = []
            action_list = []
            for batch in range(batch_size):
                obs, action, rewards, next_obs, done = experts.sample()
                obs = torch.tensor(np.array(obs)).float().unsqueeze(0).to(device)
                next_obs = torch.tensor(np.array((next_obs))).float().to(device)
                action = torch.tensor(action).long().unsqueeze(0).to(device)
                obs_list.append(obs)
                action_list.append(action)
            obs = torch.cat(obs_list)
            action = torch.cat(action_list)
            output = q_function(obs)
            loss = criterion(output,action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Model computations
            n_batch_train += 1
            # this_time = time.time()
            # print("Training batch:", n_batch_train,'total', total_step,"time", this_time - begin_time)
            # print(step)
            if (step+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, max_epochs, step+1, total_steps, loss.item()))
                writer.add_scalar('/train/loss', loss.item(), n_batch_train)
            if step % 2000 == 0:
                print("saving model....")
                torch.save(q_function.state_dict(),'./results/q_function_{}_{}.ckpt'.format(epoch, step))

    # criterion = torch.nn.MSELoss()




if __name__ == '__main__':
    main()
