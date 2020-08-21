from logging import getLogger
import os

import tqdm
import numpy as np
from sklearn.cluster import KMeans
import joblib
import minerl

logger = getLogger(__name__)


class _KMeansCacheNotFound(FileNotFoundError):
    pass


def cached_kmeans(cache_dir, env_id, n_clusters, random_state,
                  sample_by_trajectory=False, only_vector_converter=False):
    if only_vector_converter and not sample_by_trajectory:
        raise ValueError("The vector converter option must be selected with the ascending order option.")
    if cache_dir is None:  # ignore cache
        logger.info('Load dataset & do kmeans')
        kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters,
                            random_state=random_state,
                            sample_by_trajectory=sample_by_trajectory,
                            only_vector_converter=only_vector_converter)
    else:
        if only_vector_converter:
            filename = 'kmeans_vector_converter.joblib'
        else:
            if sample_by_trajectory:
                filename = 'kmeans_normal.joblib'
            else:
                filename = 'kmeans.joblib'
        filepath = os.path.join(cache_dir, env_id, f'n_clusters_{n_clusters}', f'random_state_{random_state}', filename)
        try:
            kmeans = _load_kmeans_result_cache(filepath)
            logger.info('found kmeans cache')
        except _KMeansCacheNotFound:
            logger.info('kmeans cache not found. Load dataset & do kmeans & save result as cache')
            kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters,
                                random_state=random_state,
                                sample_by_trajectory=sample_by_trajectory,
                                only_vector_converter=only_vector_converter)
            _save_kmeans_result_cache(kmeans, filepath)
    return kmeans


def _do_kmeans(env_id, n_clusters, random_state, sample_by_trajectory,
               only_vector_converter):
    logger.debug(f'loading data...')
    dat = minerl.data.make(env_id)
    if not sample_by_trajectory:
        act_vectors = []
        for ob, act, _, next_ob, _ in tqdm.tqdm(dat.batch_iter(batch_size=16, seq_len=32, num_epochs=1, preload_buffer_size=32, seed=random_state)):
            if only_vector_converter:
                if np.allclose(ob['vector'], next_ob['vector']):
                    # Ignore the case when the action does not change observation$vector.
                    continue
            act_vectors.append(act['vector'])
        acts = np.concatenate(act_vectors).reshape(-1, 64)
    else:
        episode_names = dat.get_trajectory_names()
        mem_normal = []
        mem_vc = []
        for episode_name in episode_names:
            traj = dat.load_data(episode_name)
            dn = False
            current_reward_sum = 0
            while not dn:
                ob, act, rw, next_ob, dn = next(traj)
                current_reward_sum += rw
                if np.allclose(ob['vector'], next_ob['vector']):
                    # Ignore the case when the action does not change observation$vector.
                    mem_normal.append(act['vector'])
                else:
                    mem_vc.append(act['vector'])
        if only_vector_converter:
            acts = np.array(mem_vc).reshape(-1, 64)
        else:
            acts = np.concatenate((np.array(mem_normal), np.array(mem_vc)), axis=0).reshape(-1, 64)
    logger.debug(f'loading data... done.')
    logger.debug(f'executing keamns...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(acts)
    logger.debug(f'executing keamns... done.')
    return kmeans


# def _describe_kmeans_result(kmeans):
#     result = [(obf_a, minerl.herobraine.envs.MINERL_TREECHOP_OBF_V0.unwrap_action({'vector': obf_a})) for obf_a in kmeans.cluster_centers_]
#     logger.debug(result)
#     return result


def _save_kmeans_result_cache(kmeans, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(kmeans, filepath)
    logger.info(f'saved kmeans {filepath}')


def _load_kmeans_result_cache(filepath):
    if not os.path.exists(filepath):
        raise _KMeansCacheNotFound
    logger.debug(f'loading kmeans {filepath}')
    return joblib.load(filepath)
