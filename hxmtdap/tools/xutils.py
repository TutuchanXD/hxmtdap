# 对拟合结果的一些处理工具

import os
import re
import shlex
import shutil
import tempfile
import subprocess
from copy import copy
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from corner import corner
from astropy.io import fits
from astropy.table import Table
from uncertainties import ufloat, umath

APPOINTMENT = {"Rin": "0.1 1 1 100 100"}


COMMAND_KEYWORDS = {
    "statistic",
    "method",
    "abund",
    "xsect",
    "cosmo",
    "xset",
    "systematic",
    "bayes",
    "fit",  # 添加fit等其他常见命令
    # 注意: query, error, renorm, tclout 等也可能是命令，根据需要添加
}

DATA_LOADING_KEYWORDS = {"cd", "data", "response", "backgrnd"}


def parse_xcm_file(filename: str) -> list[str]:
    """
    解析 XCM 文件，将其内容分类为数据载入命令、其他命令、忽略范围和模型定义。

    Args:
        filename: XCM 文件的路径。

    Returns:
        一个包含四个字符串元素的列表：
        [0]: 数据载入命令 (多行命令用换行符连接)。
        [1]: 其他普通命令 (多行命令用换行符连接)。
        [2]: 'ignore' 命令所在的行 (如果存在)，否则为空字符串。
        [3]: 'model' 定义及其参数行 (多行用换行符连接)。
    """
    data_loading_commands = []  # 存储数据载入命令
    other_commands = []  # 存储其他普通命令
    ignore_str = ""  # 存储 ignore 命令行的字符串
    model_lines = []  # 存储模型和参数行的列表
    in_model = False  # 标记是否在模型部分内部

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue
                first_word = parts[0]

                # 核心逻辑判断
                if first_word == "model":
                    in_model = True
                    model_lines.append(line)
                elif in_model:
                    # 检查是否是模型结束的标志
                    # 注意: 数据载入命令通常不会出现在模型定义中间，但以防万一也检查
                    if (
                        first_word == "ignore"
                        or first_word in COMMAND_KEYWORDS
                        or first_word in DATA_LOADING_KEYWORDS
                    ):
                        in_model = False
                        # 处理当前这行（结束模型部分的这行）
                        if first_word == "ignore":
                            ignore_str = line
                        elif first_word in DATA_LOADING_KEYWORDS:
                            data_loading_commands.append(line)
                        else:  # 是其他命令关键字
                            other_commands.append(line)
                    else:
                        # 继续作为模型参数行
                        model_lines.append(line)
                elif first_word == "ignore":
                    ignore_str = line
                elif first_word in DATA_LOADING_KEYWORDS:
                    # 是数据载入命令
                    data_loading_commands.append(line)
                else:
                    # 默认归类为其他普通命令
                    # (这包括 COMMAND_KEYWORDS 中的命令以及未在任何列表中的命令)
                    other_commands.append(line)

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。")
        return ["", "", "", ""]
    except Exception as e:
        print(f"读取或解析文件 '{filename}' 时发生错误: {e}")
        return ["", "", "", ""]

    # 将列表合并为字符串
    data_loading_str = "\n".join(data_loading_commands)
    other_commands_str = "\n".join(other_commands)
    model_str = "\n".join(model_lines)

    # 返回包含四个字符串的列表
    return [data_loading_str, other_commands_str, ignore_str, model_str]


def extract_data_file_paths(
    data_loading_commands_str: str, xcm_file_dir: str = None
) -> list[str]:
    """
    从包含数据载入命令的字符串中提取 .pha, .rmf, .rsp 文件的绝对路径。
    处理 'cd' 命令以解析相对路径。返回结果按解析顺序排列，并去除重复项。

    Args:
        data_loading_commands_str: 一个多行字符串，每行是一个数据载入命令
                                    (通常是 parse_xcm_file 返回列表的第一个元素)。
        xcm_file_dir: 可选参数，原始 XCM 文件所在的目录路径。
                        如果提供，相对路径将相对于此目录解析；
                        否则，相对路径将相对于当前工作目录解析 (os.getcwd())。

    Returns:
        一个包含找到的所有 .pha, .rmf, .rsp 文件绝对路径的列表，
        按解析顺序排列，并去除重复项（保留首次出现的路径）。
    """
    file_paths = []  # 存储按顺序找到的所有路径（可能包含重复）
    seen_paths = set()  # 用于快速检查路径是否已添加
    ordered_unique_paths = []  # 最终返回的有序且唯一的路径列表

    target_extensions = (".pha", ".rmf", ".rsp")

    # 确定基础目录用于解析相对路径
    if xcm_file_dir:
        current_dir = os.path.abspath(xcm_file_dir)
    else:
        current_dir = os.getcwd()
        # print("警告: 未提供 xcm_file_dir，相对路径将基于当前工作目录解析:", current_dir) # 可以取消注释此行进行调试

    lines = data_loading_commands_str.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 移除行尾分号注释
        comment_pos = line.find(";")
        if comment_pos != -1:
            line = line[:comment_pos].strip()

        try:
            parts = shlex.split(line, posix=(os.name != "nt"))
        except ValueError:
            print(f"警告: shlex 解析行失败，将使用简单分割: '{line}'")
            parts = line.split()

        if not parts:
            continue

        command = parts[0].lower()

        if command == "cd":
            if len(parts) > 1:
                new_dir_part = parts[1]
                if os.path.isabs(new_dir_part):
                    current_dir = os.path.abspath(new_dir_part)
                else:
                    current_dir = os.path.abspath(
                        os.path.join(current_dir, new_dir_part)
                    )
            # else:
            #     print(f"警告: 'cd' 命令缺少参数: '{line}'") # 可以取消注释此行进行调试

        elif command in ("data", "response", "backgrnd"):
            found_file_on_line = False
            for part in parts[1:]:
                # 检查是否是目标文件类型 (忽略大小写)
                part_lower = part.lower()
                # 确保只匹配以这些扩展名结尾的部分，避免误匹配目录名等
                is_target_file = False
                for ext in target_extensions:
                    if part_lower.endswith(ext):
                        is_target_file = True
                        break

                if is_target_file:
                    potential_file = part
                    # 解析路径
                    if os.path.isabs(potential_file):
                        abs_path = os.path.abspath(potential_file)
                    else:
                        abs_path = os.path.abspath(
                            os.path.join(current_dir, potential_file)
                        )

                    # 检查是否已经添加过这个路径
                    if abs_path not in seen_paths:
                        ordered_unique_paths.append(abs_path)
                        seen_paths.add(abs_path)
                    # 即使重复，也标记为已找到，以便跳出内循环（如果逻辑如此）
                    found_file_on_line = True
                    # 假设每行 data/resp/back 命令只指定一个主要文件
                    break  # 移除了 break，以便处理一行有多个文件的情况？不，XCM通常一行一个
            # if not found_file_on_line:
            #     print(f"警告: 在 {command} 命令中未找到指定类型的文件: '{line}'") # 可以取消注释此行进行调试

    return ordered_unique_paths


def get_timerange_from_xcm(xcmfile):
    """
    从XCM文件中获取时间范围，应该传入XCM文件的绝对路径
    返回MJD格式的时间范围
    """
    xcm_parsed = parse_xcm_file(xcmfile)
    data_parsed = xcm_parsed[0]
    phalst = extract_data_file_paths(data_parsed, os.path.dirname(xcmfile))

    from .lcutils import convert_TT2MJD

    # 提取TT时间默认使用第一个pha文件
    tstart = fits.getheader(phalst[0], 1)["TSTART"]
    tstop = fits.getheader(phalst[0], 1)["TSTOP"]
    mjdstart = convert_TT2MJD(tstart)
    mjdstop = convert_TT2MJD(tstop)

    return mjdstart, mjdstop


def get_phafile_from_xcm(xcmfile, more=False):
    """
    从XCM文件中获取phafile
    """
    try:
        oripath = os.getcwd()
        xcmfile = Path(xcmfile)
        with open(xcmfile) as f:
            contents = f.read()

        os.chdir(xcmfile.parent)
        relpath = re.findall(r"cd (.*)\s", contents)[0]
        os.chdir(relpath)
        nowpath = Path(os.getcwd())
        lefile = re.findall(r"data 1:1 (.*\.pha)", contents)[0]
        mefile = re.findall(r"data 2:2 (.*\.pha)", contents)[0]
        hefile = re.findall(r"data 3:3 (.*\.pha)", contents)[0]

        leabspath = nowpath / lefile
        meabspath = nowpath / mefile
        heabspath = nowpath / hefile

        leabspath = str(leabspath)
        meabspath = str(meabspath)
        heabspath = str(heabspath)

        os.chdir(oripath)
    except KeyboardInterrupt:
        os.chdir(oripath)

    if not more:
        return leabspath, meabspath, heabspath
    else:
        lerppath = str(nowpath / fits.getheader(leabspath, 1)["RESPFILE"])
        merppath = str(nowpath / fits.getheader(meabspath, 1)["RESPFILE"])
        herppath = str(nowpath / fits.getheader(heabspath, 1)["RESPFILE"])
        lebkgpath = str(nowpath / fits.getheader(leabspath, 1)["BACKFILE"])
        mebkgpath = str(nowpath / fits.getheader(meabspath, 1)["BACKFILE"])
        hebkgpath = str(nowpath / fits.getheader(heabspath, 1)["BACKFILE"])
        return (
            (leabspath, lerppath, lebkgpath),
            (meabspath, merppath, mebkgpath),
            (heabspath, herppath, hebkgpath),
        )


def get_mjd_from_xcm(xcmfile):
    """
    从XCM文件中获取mjd
    """
    leabsfile = get_phafile_from_xcm(xcmfile)[0]

    tstart = fits.getheader(leabsfile, 1)["TSTART"]
    tstop = fits.getheader(leabsfile, 1)["TSTOP"]

    from .lcutils import convert_TT2MJD

    mjdstart = convert_TT2MJD(tstart)
    mjdstop = convert_TT2MJD(tstop)

    return (mjdstart + mjdstop) / 2, (mjdstop - mjdstart) / 2


def open_chain(mcmcfile, modlabel=True, check=True):
    """
    返回MCMC的chain数据
    """
    if isinstance(mcmcfile, str):
        mcmchdu = Table.read(mcmcfile, 1, unit_parse_strict="silent")
        mcmcpd = mcmchdu.to_pandas()
    elif isinstance(mcmcfile, pd.DataFrame):
        mcmcpd = mcmcfile
    elif isinstance(mcmcfile, Table):
        mcmcpd = mcmcfile.to_pandas()

    if modlabel:
        mcmcpd.columns = mcmcpd.columns.str.split("__").str[0]

    if check:
        mcmcpd_std = mcmcpd.std()
        mcmcpd = mcmcpd.loc[:, (mcmcpd_std > 0).values]

    return mcmcpd


def get_nvmax(
    lineE, lineE_errL, lineE_errU, width, width_errL, width_errU, ignore_sign=False
):
    """
    根据lineE和width的误差计算nvmax，使用线性误差传递
    """
    if ignore_sign:
        lineE_errL = abs(float(lineE_errL))
        lineE_errU = abs(float(lineE_errU))
        width_errL = abs(float(width_errL))
        width_errU = abs(float(width_errU))
    else:
        lineE_errL = float(lineE_errL)
        lineE_errU = float(lineE_errU)
        width_errL = float(width_errL)
        width_errU = float(width_errU)

    def _isnan(v):
        if isinstance(v, float):
            return True
        else:
            return False

    if lineE_errL >= 0:
        lineE_vL = ufloat(
            float(f"{float(lineE):.6f}"), float(f"{float(lineE_errL):.6f}")
        )
    else:
        lineE_vL = np.nan

    if lineE_errU >= 0:
        lineE_vU = ufloat(
            float(f"{float(lineE):.6f}"), float(f"{float(lineE_errU):.6f}")
        )
    else:
        lineE_vU = np.nan

    if width_errL >= 0:
        width_vL = ufloat(
            float(f"{float(width):.6f}"), float(f"{float(width_errL):.6f}")
        )
    else:
        width_vL = np.nan

    if width_errU >= 0:
        width_vU = ufloat(
            float(f"{float(width):.6f}"), float(f"{float(width_errU):.6f}")
        )
    else:
        width_vU = np.nan

    nvmax_v = np.sqrt(lineE**2 + (width / 2) ** 2)

    # for zero-central lor
    if lineE == 0.0:
        lineE_vL, lineE_vU = ufloat(0, 0), ufloat(0, 0)

    if _isnan(lineE_vL) or _isnan(width_vL):
        nvmax_vL = np.nan
    else:
        nvmax_vL = umath.sqrt(lineE_vL**2 + (width_vL / 2) ** 2).s

    if _isnan(lineE_vU) or _isnan(width_vU):
        nvmax_vU = np.nan
    else:
        nvmax_vU = umath.sqrt(lineE_vU**2 + (width_vU / 2) ** 2).s

    return nvmax_v, nvmax_vL, nvmax_vU


def get_rms(norm, norm_errL, norm_errU):
    """
    计算rms，使用线性误差传递
    """

    def _isnan(v):
        if isinstance(v, float):
            return True
        else:
            return False

    if float(norm_errL) >= 0:
        norm_vL = ufloat(float(f"{float(norm):.6f}"), float(f"{float(norm_errL):.6f}"))
    else:
        norm_vL = np.nan
    if float(norm_errU) >= 0:
        norm_vU = ufloat(float(f"{float(norm):.6f}"), float(f"{float(norm_errU):.6f}"))
    else:
        norm_vU = np.nan

    rms_v = np.sqrt(norm)

    if _isnan(norm_vL):
        rms_vL = np.nan
    else:
        rms_vL = umath.sqrt(norm_vL).s

    if _isnan(norm_vU):
        rms_vU = np.nan
    else:
        rms_vU = umath.sqrt(norm_vU).s

    return rms_v, rms_vL, rms_vU


class LogDataResolver:

    def __init__(self, logfile: str):
        self.logfile = logfile
        self.parameters_stat = {}
        self.model_used = ""
        self.statistics = {"Statistic": None, "Bins": None}
        self.extract_data()

        self.xcmfile = None

    @staticmethod
    def from_xcmfile(xcmfile, prexcmfile=None, timeout=None):
        """
        从XCM文件生成log，返回LogDataResolver对象。
        并发安全：不修改全局CWD；不再起 multiprocessing.Process 子进程。
        通过 `subprocess.run(["xspec"], cwd=...)` 执行 XSPEC 命令脚本。
        """
        xcmfile = str(Path(xcmfile).resolve())
        xcm_dir = str(Path(xcmfile).parent.resolve())
        xcm_name = Path(xcmfile).name

        # 日志文件使用与输入同名的 .log，存放在同目录
        logfile = str(Path(xcmfile).with_suffix(".log"))
        log_name = Path(logfile).name

        # prexcmfile 若给出，也转为绝对路径；在 xspec 中用相对名（基于 cwd=xcm_dir）
        prexcm_rel = None
        if prexcmfile:
            prexcmfile = str(Path(prexcmfile).resolve())
            prexcm_rel = (
                str(Path(prexcmfile).relative_to(xcm_dir))
                if Path(prexcmfile).is_relative_to(xcm_dir)
                else prexcmfile
            )  # 若不在同目录，仍然可用绝对路径

        # 可选：从用户家目录加载 pyxspec.rc（若存在）
        user_home = Path.home()
        xspec_rcfile = user_home / ".xspec" / "pyxspec.rc"
        xspec_rc_rel = None
        if xspec_rcfile.exists():
            try:
                xspec_rc_rel = (
                    str(xspec_rcfile.relative_to(xcm_dir))
                    if xspec_rcfile.is_relative_to(xcm_dir)
                    else str(xspec_rcfile)
                )
            except Exception:
                xspec_rc_rel = str(xspec_rcfile)

        # 生成一个临时的 xspec 脚本（与输入文件同目录），避免 chdir
        # 注意：XSPEC 脚本里 '@file' 表示执行 file.xcm
        run_xcm_path = str(Path(xcm_dir) / "_run_for_log.xcm")
        script_lines = []

        # 若提供 prexcm，先设能量网格，再 restore
        if prexcm_rel:
            script_lines.append("energies 0.0001 1024. 3500 log")
            script_lines.append(f"@{prexcm_rel}")
        # 否则若家目录有 rc，就先加载
        elif xspec_rc_rel:
            script_lines.append(f"@{xspec_rc_rel}")

        # 再加载目标 xcm
        script_lines.append(f"@{xcm_name}")

        # 打开日志
        script_lines.append(f"log {log_name}")
        # 输出关键信息到日志
        script_lines += ["show all", "log none", "exit"]

        with open(run_xcm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines) + "\n")

        # 准备环境：限制并行库线程数，隔离 PFILES，避免并发互相干扰
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("HEADASNOQUERY", "1")
        pfiles_dir = str(Path(xcm_dir) / "pfiles")
        os.makedirs(pfiles_dir, exist_ok=True)
        env["PFILES"] = f"{pfiles_dir};{env.get('PFILES','.')}"
        # 若你的 HEASOFT/CALDB 需要额外环境变量，也可在这里 env[...] 指定

        # 确认 xspec 可执行存在
        xspec_prog = shutil.which("xspec")
        if not xspec_prog:
            raise RuntimeError(
                "未找到 'xspec' 可执行文件，请确认已正确初始化 HEASOFT/XSPEC 环境。"
            )

        # 运行 XSPEC：把我们生成的脚本喂给 stdin
        try:
            with open(run_xcm_path, "rb") as fin:
                subprocess.run(
                    [xspec_prog],
                    stdin=fin,
                    cwd=xcm_dir,  # << 关键：不切全局CWD，仅为本次调用指定工作目录
                    env=env,
                    stdout=subprocess.PIPE,  # 如需调试也可改为 None
                    stderr=subprocess.PIPE,
                    timeout=timeout,  # 可传入秒数避免卡死
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"XSPEC 执行失败（returncode={e.returncode}）。"
                f"\nSTDOUT:\n{e.stdout.decode('utf-8', 'ignore')}\n"
                f"\nSTDERR:\n{e.stderr.decode('utf-8', 'ignore')}"
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"XSPEC 执行超时：{run_xcm_path}")

        # 构建并返回 LogDataResolver
        loger = LogDataResolver(logfile)  # 你的现有构造应能用绝对路径
        loger.xcmfile = xcmfile
        return loger

    def extract_data(self):

        try:
            with open(self.logfile, "r") as file:
                log_data = file.read()
        except FileNotFoundError:
            return {"Error": "Log file not found"}

        # 匹配模型的正则
        model_pattern = (
            r"#?Current model list:\n[#\s]+={10,}\n#?Model\s([\w<>*\+\s]+)\sSource\sNo"
        )

        # 匹配参数的正则
        # - (\d+): 匹配参数ID.
        # - (\d+): 匹配组分ID.
        # - (\w+): 匹配组分名.
        # - (\w+): 匹配参数名.
        # - ([0-9\.E\+\-]+): 匹配可能存在的单位.
        # - ((frozen|\+/\-[ \t]|=[ \t])): 匹配参数状态.
        # - ([\n0-9\.pE\+\-\*]*): 匹配自由参数的参考误差.
        parameter_pattern = r"(\d+)[ \t]+(\d+)[ \t]+(\w+)[ \t]+(\w+)[ \t\w\-\^]+[ \t]([0-9\.E\+\-]+)[ \t]+(frozen|\+/\-[ \t]|=[ \t])([\n0-9\.pE\+\-\*/^]*)[ \t\n]*"

        # 匹配统计量的正则
        statistic_pattern = (
            r"Test\sstatistic\s:\s[\w\-]+\s+([\d\.]+)\s+using\s([\d]+)\sbins\."
        )

        # 模型提取
        model_match = re.search(model_pattern, log_data)
        self.model_used = (
            model_match.group(1).strip() if model_match else "No model found"
        )

        # 统计数据组数
        self.data_groups = len(set(re.findall(r"Data group: (\d+)", log_data)))

        # 模型列表提取
        self.components = list(re.findall(r"\b(\w+)<\d+>", self.model_used))

        # 统计量提取
        statistics = re.findall(statistic_pattern, log_data)[0]
        self.statistics = {
            "Statistic": float(statistics[0]),
            "Bins": int(statistics[1]),
        }

        # 参数提取
        parameters = re.findall(parameter_pattern, log_data)
        self.parameters_stat = {
            int(param[0]): {
                "ModelID": param[1],
                "ModelName": param[2],
                "ParamName": param[3],
                "Value": param[4],
                "Status": param[5].strip(),
                "Details": param[6].strip(),
            }
            for param in parameters
        }

        if self.data_groups > 1:
            self.paramsForSingleDG = max(self.parameters_stat.keys()) / self.data_groups
        else:
            self.data_groups = 1
            self.paramsForSingleDG = max(self.parameters_stat.keys())

        result = {
            "Model Used": self.model_used,
            "Number of Data Groups": self.data_groups,
            "Components": self.components,
            "Parameters": self.parameters_stat,
        }

        return result

    def filter_by_modelID(self, model_id):
        """
        获取某个模型的所有参数
        """
        filtered_dict = {}
        # 遍历输入字典的每一个条目
        for key, value in self.parameters_stat.items():
            # 检查子字典中的 'ModelID' 是否与输入的 modelid 匹配
            if value["ModelID"] == str(model_id):
                # 将匹配的子字典添加到结果字典中
                filtered_dict[key] = value
        return filtered_dict

    def linked_mapping(self):
        """
        暂时已弃用
        """
        self.parameters_stat_lk = copy(self.parameters_stat)
        for param_id, param_info in self.parameters_stat.items():
            if param_info["Status"] == "=":
                target_id_lst = re.findall(r"p(\d{1,})", param_info["Details"])
                for target_id in target_id_lst:
                    self.parameters_stat_lk[param_id]["Link"] = (
                        self.parameters_stat[target_id]["ModelName"],
                        self.parameters_stat[target_id]["ParamName"],
                    )

    def reconstruct_model(self, new_model: str):
        """
        基于原模型重建模型，新模型的参数会与原模型保持一致
        新模型的组分顺序需和原模型保持一致
        """

        xcm_content = [f"model {new_model}"]

        # 提取组件名称
        # 匹配新模型组件的正则
        # - ([a-zA-Z]+): 匹配组件名
        # - (?:…)?: 非捕获的可选组
        # - (\d+): 匹配组件序号
        new_model_components: list = re.findall(r"([a-zA-Z]+)(?:_(\d+))?", new_model)
        new_model_ids = {}  # 用户指定，新模型中组件对应于原参数表的组件序号
        first_instance = {}  # 如果用户未指定序号，记录第一个同名组件的序号
        parameter_map = {}  # 映射原始参数ID到新参数ID {orig_param_id: new_param_id}
        new_param_id = 1

        # 先找出新模型中每一个组件对应于原参数表的组件序号
        for name, idx in new_model_components:
            if idx:
                composite_key = (name, idx)  # 使用元组(name, idx)作为字典的键
                new_model_ids[composite_key] = idx
            else:
                if name not in first_instance:
                    for param_id, param_info in sorted(self.parameters_stat.items()):
                        if (
                            param_info["ModelName"] == name
                            and param_info["ModelID"] not in first_instance.values()
                        ):
                            composite_key = (
                                name,
                                param_info["ModelID"],
                            )  # 因为没有指定序号，所以使用元组(name, ModelID)作为字典的键
                            first_instance[composite_key] = param_info["ModelID"]
                            break
                if name in first_instance:
                    raise Exception(
                        "Please use the pattern '<component>_<id>' for the component with same name!"
                    )

        new_model_string = re.sub(r"_\d+", "", new_model)
        xcm_content = [f"model {new_model_string}"]

        # 遍历原参数表
        for param_id, param_info in sorted(self.parameters_stat.items()):

            current_key = (param_info["ModelName"], param_info["ModelID"])
            # 如果当前组件存在于新模型中
            if current_key in new_model_ids or current_key in first_instance:
                parameter_map[param_id] = (
                    new_param_id  # 映射原始参数ID到新参数ID (当前有效参数的index)
                )
                status = param_info["Status"]
                value = param_info["Value"]
                param_name = param_info["ParamName"]
            else:
                continue

            if "frozen" in status:
                line = f"{value:<10} {'-1':<8}"
            elif "=" in status:
                tied_param_ids = re.findall(r"p\d{1,3}", param_info["Details"])
                line = param_info["Details"]
                skip_line = False

                for id in tied_param_ids:
                    tied_param_id_original = int(id.strip("p"))
                    tied_param_id_new = parameter_map.get(tied_param_id_original, None)
                    if tied_param_id_new is None:
                        line = f"{value:<10}"
                        skip_line = True
                        break
                    line = line.replace(f"{id}", f"p{int(tied_param_id_new)}")

                if not skip_line:
                    line = f"={line:<8}"
            else:
                if param_name in APPOINTMENT.keys():
                    line = f"{value} {APPOINTMENT[param_name]:<8}"
                else:
                    line = f"{value:<10}"
            xcm_content.append(line)
            new_param_id += 1

        # with open("reconstructed_model.xcm", "w") as f:
        #     for line in xcm_content:
        #         f.write(line + "\n")
        return "\n".join(xcm_content)

    def stat_spec_result(self, mcmcfile=None, errange=90, get_value_from_mcmc=False):
        """
        统计能谱拟合结果，会从MCMC文件中获取参数误差
        """
        free_dict = {}
        frozen_dict = {}
        linked_dict = {}

        for index, param in self.parameters_stat.items():
            param_name = param["ParamName"]
            param_value = param["Value"]
            param_status = param["Status"]
            if "frozen" in param_status:
                frozen_dict[f"{param_name}__{index}"] = [param_value]
            elif "=" in param_status:
                linked_dict[f"{param_name}__{index}"] = [param_value]
            else:
                free_dict[f"{param_name}__{index}"] = [param_value]

        free_dict["REDUCED_CHI2"] = [
            self.statistics["Statistic"] / self.statistics["Bins"]
        ]

        if not bool(mcmcfile):
            return pd.DataFrame.from_dict(free_dict, orient="index", columns=["value"])
        else:
            mcmcchain = open_chain(mcmcfile, modlabel=False)

        efree_dict = {}
        for pname, pvalue in free_dict.items():
            if pname != "REDUCED_CHI2":
                chains = mcmcchain.loc[:, pname]
            else:
                chains = (mcmcchain.loc[:, "FIT_STATISTIC"]) / self.statistics["Bins"]
            percentileleft, percentileright, percentilemid = [
                (100 - errange) / 2,
                100 - (100 - errange) / 2,
                50,
            ]
            vL, vU, vmid = np.percentile(
                chains, [percentileleft, percentileright, percentilemid]
            )
            if get_value_from_mcmc:
                v_used = vmid
            else:
                v_used = float(pvalue[0])
            errL, errU = (v_used - vL), (vU - v_used)
            efree_dict[pname] = [v_used, errL, errU]

        efree_pd = pd.DataFrame.from_dict(
            efree_dict, orient="index", columns=["value", "errL", "errU"]
        )
        return efree_pd

    def stat_pds_result(
        self, mcmcfile=None, errange=90, nvmax=True, ext_json=None, ignore_sign=False
    ):
        """
        统计功率谱拟合结果，会从MCMC文件中读取误差
        """
        # 如果当前目录有同前缀mcmcfile会自动指定
        if not bool(mcmcfile):
            fmcmcfile = self.logfile.replace(".log", ".fits")
            if os.path.exists(fmcmcfile):
                mcmcfile = fmcmcfile

        lorentz_dict = {}
        powerlaw_dict = {}

        for index, param in self.parameters_stat.items():
            model_id = param["ModelID"]
            param_model = param["ModelName"]
            if param_model not in ["powerlaw", "lorentz"]:
                raise ValueError(f"Model {param_model} is not supported!")
            param_name = param["ParamName"]
            param_value = param["Value"]
            param_status = param["Status"]

            if param_model == "lorentz":
                if f"lorentz__{model_id}" not in lorentz_dict:
                    lorentz_pd = pd.DataFrame(
                        data=None,
                        index=[f"lorentz__{model_id}"],
                        columns=[
                            "LineE_num",
                            "LineE",
                            "LineE_errL",
                            "LineE_errU",
                            "Width_num",
                            "Width",
                            "Width_errL",
                            "Width_errU",
                            "norm_num",
                            "norm",
                            "norm_errL",
                            "norm_errU",
                        ],
                    )
                else:
                    lorentz_pd = lorentz_dict[f"lorentz__{model_id}"]

                lorentz_pd.loc[f"lorentz__{model_id}", param_name] = float(param_value)
                lorentz_pd.loc[f"lorentz__{model_id}", f"{param_name}_num"] = index
                lorentz_dict[f"lorentz__{model_id}"] = lorentz_pd

            elif param_model == "powerlaw":
                if f"powerlaw__{model_id}" not in powerlaw_dict:
                    powerlaw_pd = pd.DataFrame(
                        data=None,
                        index=[f"powerlaw__{model_id}"],
                        columns=[
                            "PhoIndex_num",
                            "PhoIndex",
                            "PhoIndex_errL",
                            "PhoIndex_errU",
                            "norm_num",
                            "norm",
                            "norm_errL",
                            "norm_errU",
                        ],
                    )
                else:
                    powerlaw_pd = powerlaw_dict[f"powerlaw__{model_id}"]

                powerlaw_pd.loc[f"powerlaw__{model_id}", param_name] = float(
                    param_value
                )
                powerlaw_pd.loc[f"powerlaw__{model_id}", f"{param_name}_num"] = index
                powerlaw_dict[f"powerlaw__{model_id}"] = powerlaw_pd

        if bool(lorentz_dict):
            lorentz_allpd = pd.concat(lorentz_dict.values(), axis=0)
        else:
            lorentz_allpd = pd.DataFrame()
        if bool(powerlaw_dict):
            powerlaw_allpd = pd.concat(powerlaw_dict.values(), axis=0)
        else:
            powerlaw_allpd = pd.DataFrame()

        if not bool(mcmcfile):
            if ext_json:
                ext_df = pd.read_json(ext_json).T
                lorentz_allpd.update(ext_df)
            return lorentz_allpd.dropna(axis=1), powerlaw_allpd.dropna(axis=1)
        else:
            mcmcpd = open_chain(mcmcfile, modlabel=False)

        # ------------ 下面获取误差 ------------
        num_columns_to_check_lorentz = [0, 4, 8]
        num_columns_to_check_powerlaw = [0, 4]

        def find_num_index(df, num_columns_to_check, model="lorentz"):
            if model == "lorentz":
                columns_to_check = num_columns_to_check_lorentz
            elif model == "powerlaw":
                columns_to_check = num_columns_to_check_powerlaw
            # 迭代检查指定列
            for col in columns_to_check:
                # 使用 np.where 查找值的索引
                found_indices = np.where(df.iloc[:, col] == num_columns_to_check)[0]
                if len(found_indices) > 0:  # 如果找到了值
                    row_index = found_indices[0]  # 取第一个（也是唯一的）索引
                    return [row_index, col]  # 返回 [行, 列]
            # 如果值未找到，返回 None
            return None

        for cparam in mcmcpd.columns:
            chains = mcmcpd[cparam]
            percentileleft, percentileright = [
                (100 - errange) / 2,
                100 - (100 - errange) / 2,
            ]
            vL, vU = np.percentile(chains, [percentileleft, percentileright])
            if cparam == "FIT_STATISTIC":
                continue
            else:
                cparam_name, cparam_num = cparam.split("__")

            # 查找参数所在行列
            if cparam_name in ["LineE", "Width"]:
                index = find_num_index(lorentz_allpd, int(cparam_num), model="lorentz")
                if index is None:
                    raise ValueError(f"Can not find {cparam} in {self.logfile}")
                else:
                    now_model = "lorentz"
                    row, col = index
            elif cparam_name in ["PhoIndex"]:
                index = find_num_index(
                    powerlaw_allpd, int(cparam_num), model="powerlaw"
                )
                if index is None:
                    raise ValueError(f"Can not find {cparam} in {self.logfile}")
                else:
                    now_model = "powerlaw"
                    row, col = index
            elif cparam_name in ["norm"]:
                index = find_num_index(lorentz_allpd, int(cparam_num), model="lorentz")
                if index is None:
                    index = find_num_index(
                        powerlaw_allpd, int(cparam_num), model="powerlaw"
                    )
                    if index is None:
                        raise ValueError(f"Can not find {cparam} in {self.logfile}")
                    else:
                        now_model = "powerlaw"
                        row, col = index
                else:
                    now_model = "lorentz"
                    row, col = index
            else:
                raise ValueError(f"Can not find {cparam} in {self.logfile}")

            if now_model == "lorentz":
                value = float(lorentz_allpd.iloc[row, col + 1])
                errL, errU = (value - vL), (vU - value)
                lorentz_allpd.iloc[row, col + 2] = errL
                lorentz_allpd.iloc[row, col + 3] = errU
            elif now_model == "powerlaw":
                value = float(powerlaw_allpd.iloc[row, col + 1])
                errL, errU = (value - vL), (vU - value)
                powerlaw_allpd.iloc[row, col + 2] = errL
                powerlaw_allpd.iloc[row, col + 3] = errU

        if ext_json:
            ext_df = pd.read_json(ext_json).T
            lorentz_allpd.update(ext_df)

        if not nvmax:
            return lorentz_allpd, powerlaw_allpd

        # ------------ 下面判断类型 ------------

        for index in lorentz_allpd.index.to_list():
            lineE, lineE_errL, lineE_errU = (
                lorentz_allpd.loc[index, "LineE"],
                lorentz_allpd.loc[index, "LineE_errL"],
                lorentz_allpd.loc[index, "LineE_errU"],
            )
            width, width_errL, width_errU = (
                lorentz_allpd.loc[index, "Width"],
                lorentz_allpd.loc[index, "Width_errL"],
                lorentz_allpd.loc[index, "Width_errU"],
            )
            norm, norm_errL, norm_errU = (
                lorentz_allpd.loc[index, "norm"],
                lorentz_allpd.loc[index, "norm_errL"],
                lorentz_allpd.loc[index, "norm_errU"],
            )

            # 如果该lor不是零心的，且lineE的误差没有计算
            if (np.isnan(lineE_errL) or np.isnan(lineE_errU)) and (not lineE == 0.0):

                nvmax, _, _ = get_nvmax(
                    lineE, 0, 0, width, 0, 0, ignore_sign=ignore_sign
                )
                rms, rms_errL, rms_errU = get_rms(norm, norm_errL, norm_errU)
                lorentz_allpd.loc[index, "nvmax"] = nvmax
                lorentz_allpd.loc[index, "nvmax_errL"] = np.nan
                lorentz_allpd.loc[index, "nvmax_errU"] = np.nan
                lorentz_allpd.loc[index, "rms"] = rms
                lorentz_allpd.loc[index, "rms_errL"] = rms_errL
                lorentz_allpd.loc[index, "rms_errU"] = rms_errU

                # 如果是零心
                if lineE == 0:
                    lorentz_allpd.loc[index, "comptype"] = "flat_noise"
                    continue
                # 如果非零心
                else:
                    if np.isclose(lineE, nvmax, rtol=1e-2):
                        lorentz_allpd.loc[index, "comptype"] = "QPO"
                    elif np.isclose(lineE, nvmax, rtol=1e-1):
                        lorentz_allpd.loc[index, "comptype"] = "peak_noise"
                    else:
                        lorentz_allpd.loc[index, "comptype"] = "flat_noise"

            # 如果lineE的误差有计算
            else:
                nvmax, nvmax_errL, nvmax_errU = get_nvmax(
                    lineE,
                    lineE_errL,
                    lineE_errU,
                    width,
                    width_errL,
                    width_errU,
                    ignore_sign=ignore_sign,
                )
                rms, rms_errL, rms_errU = get_rms(norm, norm_errL, norm_errU)
                lorentz_allpd.loc[index, "nvmax"] = nvmax
                lorentz_allpd.loc[index, "nvmax_errL"] = nvmax_errL
                lorentz_allpd.loc[index, "nvmax_errU"] = nvmax_errU
                lorentz_allpd.loc[index, "rms"] = rms
                lorentz_allpd.loc[index, "rms_errL"] = rms_errL
                lorentz_allpd.loc[index, "rms_errU"] = rms_errU

                if np.isclose(lineE, nvmax, rtol=1e-2):
                    lorentz_allpd.loc[index, "comptype"] = "QPO"
                elif np.isclose(lineE, nvmax, rtol=1e-1):
                    lorentz_allpd.loc[index, "comptype"] = "peak_noise"
                else:
                    lorentz_allpd.loc[index, "comptype"] = "flat_noise"

        return lorentz_allpd, powerlaw_allpd


def filter_bln(
    pdsres,
    comptypes=["flat_noise", "peak_noise"],
    columns=["nvmax", "rms", "comptype"],
    ignore_index=None,
    nums=4,
    multi=False,
):
    """
    返回BLN拟合结果

    参数：
        pdsres (pd.DataFrame): 输入的 DataFrame。
        comptypes (list): 需要筛选的 comptype 值列表，默认 ["flat_noise", "peak_noise"]。
        columns (list): 模糊匹配列名的关键字列表。
        nums (int): 如果筛选后行数大于 nums，则保留 Width 列值最大的 nums 行，默认 4。
        multi (bool): 是否返回多层索引格式，默认 False。
        ignore_index (int): 如果不为 None 且为 int，则忽略原始 DataFrame 中第 ignore_index 行，默认 None。

    返回：
        pd.DataFrame: 处理后的结果。
    """
    # 忽略指定行
    if ignore_index is not None and isinstance(ignore_index, int):
        if ignore_index < 0 or ignore_index >= len(pdsres):
            raise IndexError(f"ignore_index 超出范围: {ignore_index}")
        pdsres = pdsres.drop(pdsres.index[ignore_index])

    filtered = pdsres[pdsres["comptype"].isin(comptypes)]

    if nums is not None and len(filtered) > nums:
        if "Width" not in filtered.columns:
            raise ValueError("DataFrame 中缺少 'Width' 列，无法根据其值进行筛选。")
        filtered["Width"] = filtered["Width"].astype(float)
        filtered = filtered.nlargest(nums, "Width")  # 按 Width 列降序选取前 nums 行

    sorted = filtered.sort_values(by="nvmax", ascending=True)  # .reset_index(drop=True)

    # 模糊匹配列名
    matched_columns = [col for col in sorted.columns for key in columns if key in col]
    matched_columns = list(dict.fromkeys(matched_columns))
    sorted = sorted[matched_columns]

    # 多层索引
    if multi:
        multi_index = pd.MultiIndex.from_product(
            [sorted.index, sorted.columns], names=["row", "column"]
        )
        flattened_data = sorted.values.flatten()
        sorted = pd.DataFrame([flattened_data], columns=multi_index)

    return sorted


def separate_singel_result(xcmfile, plotmod: Literal["uf", "euf"] = "euf"):
    """
    输入xcm文件的绝对路径
    """
    import xspec

    dirpath = os.path.dirname(xcmfile)
    xcmfilename = os.path.basename(xcmfile)
    xcmfileprefix = xcmfilename.split(".")[0]
    now_path = os.getcwd()
    os.chdir(dirpath)

    xspec.Xset.restore(xcmfilename)
    xspec.AllModels.setEnergies("0.001 1024. 3500 log")
    logfile = f"{xcmfileprefix}.log"
    xspec.Xset.openLog(logfile)
    xspec.AllData.show()
    xspec.AllModels.show()
    xspec.Fit.show()
    xspec.Xset.closeLog()

    modelObj = xspec.AllModels(1)
    modellst = modelObj.componentNames

    xspec.Plot.device = "/null"
    xspec.Plot.xAxis = "keV"
    xspec.Plot.xLog = True
    xspec.Plot(plotmod)
    plotData = {
        "x": xspec.Plot.x(),
        "y": xspec.Plot.y(),
        "xerr": xspec.Plot.xErr(),
        "yerr": xspec.Plot.yErr(),
        "model": xspec.Plot.model(),
    }
    xspec.Plot("chi")
    plotData["chi2"] = xspec.Plot.y()

    # 按顺序重建
    log_resolver = LogDataResolver(logfile)
    for model in modellst:
        tmpxcm = log_resolver.reconstruct_model(model)
        with tempfile.NamedTemporaryFile(mode="w+t", delete=True) as f:
            f.write(tmpxcm)
            f.write("\n")
            f.flush()
            xspec.Xset.restore(f.name)
        xspec.Plot(plotmod)
        plotData[model] = xspec.Plot.model()

    os.chdir(now_path)

    return pd.DataFrame(plotData)


def generate_mcmc_chain(xspobj, length=30000, burn=30000, mcmcfile="mcmcfile.fits"):
    """
    不推荐使用，pyxspec无法设置并行
    """
    xspobj.AllChains.clear()

    with tempfile.NamedTemporaryFile(mode="w+t", delete=True) as temp:
        temp.write(f"chain length {length}\n")
        temp.write(f"chain burn {burn}\n")
        temp.write(f"chain walker 100\n")
        temp.write(f"chain run {mcmcfile}\n")
        temp.flush()
        xspobj.Xset.restore(temp.name)


# TODO 待完成
def generate_mcmc_chain2(
    xcmfile, length=50000, burn=50000, walker=200, mcmcfile="mcmcfile.fits"
):
    """
    生成chain，但是使用screenutil并行
    """
    try:
        from screenutils import Screen
    except ImportError:
        raise ImportError("Please install screenutil before using this function")


def generate_corner_plot(mcmcfile, percentiles=(0.05, 0.5, 0.95)):
    """
    返回corner图
    """
    mcmcpd = open_chain(mcmcfile)

    import matplotlib as mpl

    with mpl.rc_context():
        corner_plot = corner(
            data=mcmcpd.values,
            labels=mcmcpd.columns,
            quantiles=percentiles,
            show_titles=True,
            title_kwargs={
                "fontsize": 8,
            },
        )
        return corner_plot


def convert_data_to_xspec(x, y, savepath, fileprefix, xerr=None, yerr=None):
    """
    将数据转换为xspec可用的格式,
    """
    x = np.array(x)
    y = np.array(y)
    if xerr is None:
        xerr = np.zeros_like(x)
    else:
        xerr = np.array(xerr)
    if yerr is None:
        yerr = np.zeros_like(y)
    else:
        yerr = np.array(yerr)

    new_array = np.array([x - xerr, x + xerr, 2 * xerr * y, 2 * xerr * yerr]).T
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False, suffix=".txt") as f:
        np.savetxt(f.name, new_array)

    params = {
        "infile": f.name,
        "phafile": f"{savepath}/{fileprefix}.pha",
        "rspfile": f"{savepath}/{fileprefix}.rsp",
        "clobber": "yes",
    }

    from ..core.execute import gen_cmd_string, CommandExecutor

    cmd_string = gen_cmd_string("flx2xsp", params, ktype="keyword")
    print(cmd_string)
    runner = CommandExecutor()
    runner.run(cmd_string)

    return f"{savepath}/{fileprefix}.pha", f"{savepath}/{fileprefix}.rsp"
