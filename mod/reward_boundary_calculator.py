from logging import getLogger
import os

import tqdm
import numpy as np
import joblib
import minerl

logger = getLogger(__name__)


class _CacheNotFound(FileNotFoundError):
    pass


def cached_reward_boundary(cache_dir, env_id, n_groups, random_state):
    if cache_dir is None:  # ignore cache
        logger.info('Load dataset & calculate boundaries')
        boundaries = _calc_boundaries(env_id=env_id, n_groups=n_groups,
                                      random_state=random_state)
    else:
        filename = 'reward_boundaries.joblib'
        filepath = os.path.join(cache_dir, env_id, f'n_groups_{n_groups}', f'random_state_{random_state}', filename)
        try:
            boundaries = _load_result_cache(filepath)
            logger.info('found reward boundary cache')
        except _CacheNotFound:
            logger.info('boundary cache not found. Load dataset & calculate boundaries & save result as cache')
            boundaries = _calc_boundaries(env_id=env_id, n_groups=n_groups,
                                          random_state=random_state)
            _save_result_cache(boundaries, filepath)
    logger.debug(boundaries)
    return boundaries


def _calc_boundaries(env_id, n_groups, random_state):
    logger.debug(f'loading data...')
    dat = minerl.data.make(env_id)
    episode_names = dat.get_trajectory_names()
    all_rewards = []
    for episode_name in episode_names:
        current_sum = 0
        done = False
        loader = dat.load_data(episode_name)
        while not done:
            _, _, r, _, done = next(loader)
            all_rewards.append(current_sum)
            current_sum += r

    all_rewards = np.array(sorted(all_rewards))
    distrib = []
    prev, cnt = 0, 0
    for reward in all_rewards:
        if prev < reward:
            distrib.append((prev, cnt))
            prev, cnt = reward, 0
        cnt += 1
    distrib.append((prev, cnt))

    # Find boundaries
    def separate(distrib, max_region_size):
        region_size = None
        boundaries = []
        for rew, num in distrib:
            if region_size is None or region_size + num > max_region_size:
                region_size = num
                boundaries.append(rew)
            else:
                region_size += num
        return boundaries

    cand_min = 0
    cand_max = len(all_rewards)
    while cand_min < cand_max:
        half = int((cand_min + cand_max) / 2)
        if len(separate(distrib, half)) > n_groups:
            cand_min = half + 1
        else:
            cand_max = half

    return separate(distrib, cand_min)[1:]


# def _describe_kmeans_result(kmeans):
#     result = [(obf_a, minerl.herobraine.envs.MINERL_TREECHOP_OBF_V0.unwrap_action({'vector': obf_a})) for obf_a in kmeans.cluster_centers_]
#     logger.debug(result)
#     return result


def _save_result_cache(boundaries, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(boundaries, filepath)
    logger.info(f'saved boundaries {filepath}')


def _load_result_cache(filepath):
    if not os.path.exists(filepath):
        raise _CacheNotFound
    logger.debug(f'loading boundaries {filepath}')
    return joblib.load(filepath)
