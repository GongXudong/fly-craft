from imitation.data.types import TransitionsMinimal, Transitions, DictObs
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import sys
from copy import deepcopy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from flycraft.env import FlyCraftEnv
from flycraft.utils.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from flycraft.utils.load_config import load_config


def load_data_from_csv_files(
        data_dir: Path, 
        cache_data: bool,
        cache_data_dir: Path,
        trajectory_save_prefix: str="traj",
        env_config_file: Path=PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json",
        my_logger: logging.Logger=None, 
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:
    """加载数据，并根据比例划分训练集、验证集、测试集。

    返回的obs的类型(一个大号的字典，其value是整合了所有batch对应key的value数组)：{
        "observation": np.ndarray (batch_size, observation_shape),
        "achieved_goal": np.ndarray (batch_size, goal_shape),
        "desired_goal": np.ndarray (batch_size, goal_shape)
    }

    Args:
        data_dir (str, optional): 数据存储目录. Defaults to DATA_DIR.
        my_logger (logging.Logger, optional): 日志器. Defaults to None.
        train_size (float, optional): 训练集比例. Defaults to 0.9.
        validation_size (float, optional): 验证集比例. Defaults to 0.05.
        test_size (float, optional): 测试集比例. Defaults to 0.05.
        shuffle (bool, optional): 是否打乱数据集顺序. Defaults to True.

    Returns:
        Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]: 训练集、验证集、测试集
    """
    assert np.allclose([train_size + validation_size + test_size], [1.0]), "训练集、验证集、测试集的比例之和必须为1！"
    
    start_time = time()
    res_file = data_dir / "res.csv"
    res_df = pd.read_csv(res_file)

    obs = []
    acts = []
    infos = []

    traj_cnt = 0
    traj_file_cnt = 0
    transitions_cnt = 0
    
    for index, row in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        target_v, target_mu, target_chi, cur_length = row["v"], row["mu"], row["chi"], row["length"]   # TODO: 注意，有的res.csv保存了序号，每保存的话，使用row[:4]
        if cur_length > 0:
            # 能够生成轨迹的目标速度矢量
            cur_filename = f"{trajectory_save_prefix}_{int(target_v)}_{int(target_mu)}_{int(target_chi)}.csv"
            cur_file_path = data_dir / cur_filename
            transitions_cnt += cur_length
            traj_cnt += 1
            if cur_file_path.exists():
                if my_logger is not None:
                    my_logger.info(f"process file: {cur_filename}")
                else:
                    print(f"process file: {cur_filename}")
                traj_file_cnt += 1
                cur_traj = pd.read_csv(cur_file_path.absolute())

                # 9.960(s) for 13 files with pd.iterrows
                # for index, row in cur_traj.iterrows():
                #     obs.append([*row[1:9], target_v, target_mu, target_chi])
                #     acts.append([*row[9:12]])
                #     infos.append(None)

                # 1.742(s) for 13 files without pd.iterrows
                obs.extend([{
                    "observation": np.array(item[0:8], dtype=np.float32),
                    "achieved_goal": np.array(item[3:6], dtype=np.float32),
                    "desired_goal": np.array(item[8:11], dtype=np.float32)
                } for item in zip(
                    cur_traj['s_phi'].tolist(),
                    cur_traj['s_theta'].tolist(),
                    cur_traj['s_psi'].tolist(),
                    cur_traj['s_v'].tolist(),
                    cur_traj['s_mu'].tolist(),
                    cur_traj['s_chi'].tolist(),
                    cur_traj['s_p'].tolist(),
                    cur_traj['s_h'].tolist(),
                    [target_v] * cur_traj.count()['time'],
                    [target_mu] * cur_traj.count()['time'],
                    [target_chi] * cur_traj.count()['time'],
                )])
                acts.extend(zip(
                    cur_traj['a_p'].tolist(),
                    cur_traj['a_nz'].tolist(),
                    cur_traj['a_pla'].tolist(),
                ))
                infos.extend([None] * cur_traj.count()['time'])

    # 数据标准化. 这里的标准化最耗时.
    origin_env = FlyCraftEnv(config_file=env_config_file)
    scaled_obs_env = ScaledObservationWrapper(origin_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    scaled_obs = np.array([scaled_obs_env.scale_state(item) for item in obs])
    scaled_acts = np.array([scaled_act_env.scale_action(np.array(item)) for item in acts])

    if cache_data:
    # 缓存标准化后的数据
        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        np.save(str((cache_data_dir / "normalized_obs").absolute()), scaled_obs)
        np.save(str((cache_data_dir / "normalized_acts").absolute()), scaled_acts)
        np.save(str((cache_data_dir / "infos").absolute()), np.array(infos))

    # 输出统计量
    if my_logger is not None:
        my_logger.info(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        my_logger.info(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        my_logger.info(f"process time: {time() - start_time}(s).")
    else:
        print(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        print(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        print(f"process time: {time() - start_time}(s).")

    # 训练集、验证集、测试集划分
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos = train_test_split(
        scaled_obs, scaled_acts, np.array(infos), 
        train_size=train_size, 
        test_size=validation_size + test_size, 
        shuffle=shuffle,
        random_state=0,  # 保证每次得到的结果是一样的
    )

    validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos = train_test_split(
        tmp_data, tmp_labels, tmp_infos,
        train_size=validation_size/(validation_size + test_size),
        test_size=test_size/(validation_size + test_size),
        shuffle=shuffle,
        random_state=0,
    )
    print(f"划分集合后总时间：{time() - start_time}(s).")
    
    # return (
    #     TransitionsMinimal(obs=DictObs.from_obs_list(train_data), acts=train_labels, infos=train_infos),
    #     TransitionsMinimal(obs=DictObs.from_obs_list(validation_data), acts=validation_labels, infos=validation_infos),
    #     TransitionsMinimal(obs=DictObs.from_obs_list(test_data), acts=test_labels, infos=test_infos)
    # )

    train_obs = DictObs.from_obs_list(train_data)
    validation_obs = DictObs.from_obs_list(validation_data)
    test_obs = DictObs.from_obs_list(test_data)

    return (
        Transitions(obs=train_obs, acts=train_labels, infos=train_infos, next_obs=deepcopy(train_obs), dones=np.array([False]*len(train_infos))),
        Transitions(obs=validation_obs, acts=validation_labels, infos=validation_infos, next_obs=deepcopy(validation_obs), dones=np.array([False]*len(validation_infos))),
        Transitions(obs=test_obs, acts=test_labels, infos=test_infos, next_obs=deepcopy(test_obs), dones=np.array([False]*len(test_infos)))
    )

def load_data_from_cache(
        data_cache_dir: Path,
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:

    scaled_obs = np.load(str(data_cache_dir / "normalized_obs.npy"), allow_pickle=True)
    scaled_acts = np.load(str(data_cache_dir / "normalized_acts.npy"))
    infos = np.load(str(data_cache_dir / "infos.npy"), allow_pickle=True)

    # 训练集、验证集、测试集划分
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos = train_test_split(
        scaled_obs, scaled_acts, np.array(infos), 
        train_size=train_size, 
        test_size=validation_size + test_size, 
        shuffle=shuffle,
        random_state=0,  # 保证每次得到的结果是一样的
    )

    validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos = train_test_split(
        tmp_data, tmp_labels, tmp_infos,
        train_size=validation_size/(validation_size + test_size),
        test_size=test_size/(validation_size + test_size),
        shuffle=shuffle,
        random_state=0,
    )

    # return (
    #     TransitionsMinimal(obs=DictObs.from_obs_list(train_data), acts=train_labels, infos=train_infos),
    #     TransitionsMinimal(obs=DictObs.from_obs_list(validation_data), acts=validation_labels, infos=validation_infos),
    #     TransitionsMinimal(obs=DictObs.from_obs_list(test_data), acts=test_labels, infos=test_infos)
    # )

    train_obs = DictObs.from_obs_list(train_data)
    validation_obs = DictObs.from_obs_list(validation_data)
    test_obs = DictObs.from_obs_list(test_data)

    return (
        Transitions(obs=train_obs, acts=train_labels, infos=train_infos, next_obs=deepcopy(train_obs), dones=np.array([False]*len(train_infos))),
        Transitions(obs=validation_obs, acts=validation_labels, infos=validation_infos, next_obs=deepcopy(validation_obs), dones=np.array([False]*len(validation_infos))),
        Transitions(obs=test_obs, acts=test_labels, infos=test_infos, next_obs=deepcopy(test_obs), dones=np.array([False]*len(test_infos)))
    )

if __name__ =="__main__":
    data_dir = PROJECT_ROOT_DIR.parent.parent / "fly-craft-datasets" / "data" / "10hz_10_5_5_v1"
    cache_data = True
    cache_data_dir = PROJECT_ROOT_DIR.parent.parent / "fly-craft-datasets" / "cache" / "10hz_10_5_5_v1"
    
    print("Data dir: ", data_dir)
    # train_trans, validation_trans, test_trans = load_data_from_csv_files(
    #     data_dir=data_dir,
    #     cache_data=cache_data,
    #     cache_data_dir=cache_data_dir,
    #     trajectory_save_prefix="traj",
    #     train_size=0.96,
    #     validation_size=0.02,
    #     test_size=0.02
    # )

    train_trans, validation_trans, test_trans = load_data_from_cache(
        data_cache_dir=cache_data_dir,
        shuffle=True
    )

    print(type(train_trans.obs), train_trans.obs)
    
    print(type(train_trans.obs), type(train_trans.acts), type(train_trans.infos))
    print(train_trans.obs.shape, train_trans.acts.shape, train_trans.infos.shape)
    print(validation_trans.obs.shape, validation_trans.acts.shape, validation_trans.infos.shape)
    print(test_trans.obs.shape, test_trans.acts.shape, test_trans.infos.shape)
