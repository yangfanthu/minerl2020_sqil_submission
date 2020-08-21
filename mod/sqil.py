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

logger = logging.getLogger(__name__)


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
    parser.add_argument('--frame-stack', type=int, default=4, help='Number of frames stacked (None for disable).')

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

    # create & wrap env
    def wrap_env_partial(env, test):
        randomize_action = False  # test and args.noisy_net_sigma is None
        if args.dual_kmeans:
            action_choices_normal = kmeans_normal.cluster_centers_
            action_choices_vector_converter = kmeans_vector_converter.cluster_centers_  # noqa
        else:
            action_choices_normal = kmeans.cluster_centers_
            action_choices_vector_converter = None
        wrapped_env = wrap_env(
            env=env, test=test,
            env_id=args.env,
            monitor=args.monitor, outdir=args.outdir,
            frame_skip=args.frame_skip,
            gray_scale=args.gray_scale, frame_stack=args.frame_stack,
            randomize_action=randomize_action, eval_epsilon=args.eval_epsilon,
            action_choices=action_choices_normal,
            action_choices_vector_converter=action_choices_vector_converter,
            append_reward_channel=(args.option_n_groups > 1))
        return wrapped_env
    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')
    core_env = gym.make(args.env)
    # training env
    env = wrap_env_partial(env=core_env, test=False)
    # env.seed(int(train_seed))  # TODO: not supported yet
    # evaluation env
    eval_env = wrap_env_partial(env=core_env, test=True)
    # env.seed(int(test_seed))  # TODO: not supported yet (also requires `core_eval_env = gym.make(args.env)`)

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # 8,000,000 frames = 1333 episodes if we count an episode as 6000 frames,
    # 8,000,000 frames = 1000 episodes if we count an episode as 8000 frames.
    maximum_frames = args.steps
    if args.frame_skip is None:
        steps = maximum_frames
        eval_interval = args.eval_interval
    else:
        steps = maximum_frames // args.frame_skip
        eval_interval = args.eval_interval // args.frame_skip

    agent = get_agent(
        n_actions=env.action_space.n, arch=args.arch, n_input_channels=env.observation_space.shape[0],
        # noisy_net_sigma=args.noisy_net_sigma,
        final_epsilon=args.final_epsilon,
        final_exploration_frames=args.final_exploration_frames, explorer_sample_func=env.action_space.sample,
        lr=args.lr, adam_eps=args.adam_eps,
        # prioritized=args.prioritized,
        steps=steps, update_interval=args.update_interval,
        replay_capacity=args.replay_capacity, num_step_return=args.num_step_return,
        gpu=args.gpu, gamma=args.gamma, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval, clip_delta=args.clip_delta,
        batch_accumulator=args.batch_accumulator, expert_dataset=experts,
        exp_reward_scale=args.exp_reward_scale, experience_lambda=args.experience_lambda,
        reward_boundaries=boundaries, reward_channel_scale=reward_channel_scale,
    )

    if args.load:
        agent.load(args.load)

    # experiment
    if args.demo:
        eval_stats = pfrl.experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None, eval_n_episodes=args.eval_n_runs, eval_interval=eval_interval,
            outdir=args.outdir, eval_env=eval_env, save_best_so_far_agent=True,
        )

    env.close()
    eval_env.close()


def get_agent(
        n_actions, arch, n_input_channels,
        # noisy_net_sigma,
        final_epsilon, final_exploration_frames, explorer_sample_func,
        lr, adam_eps,
        # prioritized,
        steps, update_interval, replay_capacity, num_step_return,
        gpu, gamma, replay_start_size, target_update_interval, clip_delta, batch_accumulator,
        expert_dataset, exp_reward_scale, experience_lambda, reward_boundaries, reward_channel_scale,
):
    # Q function
    q_func = parse_arch(arch, n_actions, n_input_channels=n_input_channels,
                        reward_boundaries=reward_boundaries,
                        reward_channel_scale=reward_channel_scale)

    # explorer
    explorer = pfrl.explorers.Boltzmann(1.0)

    # Use the Nature paper's hyperparameters
    # opt = optimizers.RMSpropGraves(lr=lr, alpha=0.95, momentum=0.0, eps=1e-2)
    opt = torch.optim.Adam(q_func.parameters(), lr, eps=adam_eps)  # NOTE: mirrors DQN implementation in MineRL paper

    # Select a replay buffer to use
    rbuf = pfrl.replay_buffers.ReplayBuffer(replay_capacity, num_step_return)

    # build agent
    def phi(x):
        # observation -> NN input
        return np.asarray(x)
    agent = SQIL(
        q_func, opt, rbuf, gpu=gpu, gamma=gamma, explorer=explorer, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta, update_interval=update_interval,
        batch_accumulator=batch_accumulator, phi=phi, expert_dataset=expert_dataset,
        reward_scale=exp_reward_scale, experience_lambda=experience_lambda,
        reward_boundaries=reward_boundaries)

    return agent


if __name__ == '__main__':
    main()
