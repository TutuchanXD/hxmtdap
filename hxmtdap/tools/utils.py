# 这个模块将存放一些通用功能函数
import os
import re

import numpy as np
import pandas as pd
from collections import deque


def get_expID(path, check_basename=False):
    """
    提取路径中的曝光ID；额外参数check_basename检查是否以expID为基路径
    """
    expID = re.findall(r"(P\d{8,15})", path)
    if expID:
        basepath = os.path.basename(path)
        whether_basepath = True if re.search(r"(P\d{8,15})", basepath) else False
    else:
        raise Exception(f"The expID is not found in {path}")
    if check_basename:
        return expID[-1], whether_basepath
    else:
        return expID[-1]


def get_expID_path(path):
    """
    提取长路径中直到符合曝光ID的路径
    """
    exposure_id_pattern = r"P\d{12}[\d\-]*"

    # 将路径按分隔符分割
    path_parts = path.split(os.sep)

    # 遍历路径的每一部分，寻找匹配曝光ID的层
    for i, part in enumerate(path_parts):
        if re.match(exposure_id_pattern, part):
            # 找到匹配的曝光ID，返回从根到这一层的路径
            return os.sep.join(path_parts[: i + 1])

    # 如果没有找到匹配的曝光ID，返回空
    return None


def find_expID_path(expID, dirpath=None, depth=-1):
    """
    在指定目录下查找包含特定曝光ID的路径

    参数:
        expID: 曝光ID
        dirpath: 起始搜索目录，默认为当前目录
        depth: 搜索深度，默认为-1表示无限深度
            depth=0 表示只搜索起始目录
            depth=1 表示搜索起始目录及其直接子目录
            depth=2 表示搜索到"孙子"目录

    返回:
        找到的路径，未找到则返回None
    """
    # 如果dirpath为None，使用当前目录
    if dirpath is None:
        dirpath = os.getcwd()

    # 确保dirpath是有效的目录路径
    if not os.path.isdir(dirpath):
        raise ValueError(f"指定的路径'{dirpath}'不是一个有效的目录")

    # 使用队列实现广度优先搜索
    queue = deque([(dirpath, 0)])  # (路径, 当前深度)
    while queue:
        current_path, current_depth = queue.popleft()
        # 如果达到指定深度限制，跳过此路径
        if depth != -1 and current_depth > depth:
            continue
        try:
            # 获取当前目录下的所有项目
            items = os.listdir(current_path)
            # 首先检查当前层的目录
            for item in items:
                full_path = os.path.join(current_path, item)
                if os.path.isdir(full_path) and expID in item:
                    return full_path
            # 将子目录添加到队列中
            for item in items:
                full_path = os.path.join(current_path, item)
                if os.path.isdir(full_path):
                    queue.append((full_path, current_depth + 1))
        except (PermissionError, OSError):
            # 忽略没有权限访问的目录
            continue


def find_nearest(array, value):
    """
    找到数组中最接近指定值的元素值
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def energy_to_pi(energy, instrument, logger=None):
    """
    输入能量，返回对应的PI

    - energy : 能量
    - instrument : LE、ME、HE
        LE:
        PI = int(1536*(energy-0.1)/13)
        ME:
        PI = 1024*(energy-3)/60
        HE:
        PI = 256(E-15)/370
    """
    if not energy:
        return
    if not logger:
        warning = print
    else:
        warning = logger.warning

    energy = float(energy)
    if instrument == "HE":
        if ((energy < 27) or (energy > 250)) and logger:
            warning(
                f"The HE instrument cannot produce a reliable background light curve of {energy} keV \
                        (27 -- 250 keV are recommened)"
            )
        PI = int(256 * (energy - 15) / 370)
    elif instrument == "ME":
        if (energy < 10) or (energy > 35):
            warning(
                f"The ME instrument cannot produce a reliable background light curve of {energy} keV \
                        (10 -- 35 keV are recommened)"
            )
        PI = int(1024 * (energy - 3) / 60)
    elif instrument == "LE":
        if (energy < 1) or (energy > 10):
            warning(
                f"The LE instrument cannot produce a reliable background light curve of {energy} keV \
                        (1 -- 10 keV are recommened)"
            )
        PI = int(1536 * (energy - 0.1) / 13)
    return PI


def pi_to_energy(pi, instrument):
    """
    输入PI，返回对应的能量

    - pi : PI 值
    - instrument : LE、ME、HE
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data", "pi_to_energy.csv")
    mapper = pd.read_csv(csv_path)
    mapper = mapper[mapper.Dti == instrument]
    nearest = find_nearest(mapper.loc[:, "Pi"], float(pi))

    # 获取所有与 nearest 匹配的行
    matched_rows = mapper[mapper.Pi == nearest]
    energies = matched_rows.Energy.values

    # 返回所有值的列表，并尽可能将能量转换为整数
    return [int(e) if e == int(e) else e for e in energies]


def find_most_recent_file_in_directories(*directories):
    """
    查找多个目录中所有子文件中修改日期最近的文件

    - directories: 目录路径列表
    """
    all_files = []

    for directory in directories:
        # 遍历目录及其子目录中的所有文件
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    if not all_files:
        return None

    # 找出修改日期最近的文件
    most_recent_file = max(all_files, key=os.path.getmtime)

    return most_recent_file
