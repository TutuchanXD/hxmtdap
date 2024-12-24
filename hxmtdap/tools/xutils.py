# 对拟合结果的一些处理工具

import os
import re
import tempfile
from copy import copy
from pathlib import Path
from typing import Literal
from multiprocessing import Process

import numpy as np
import pandas as pd
from corner import corner
from astropy.io import fits
from astropy.table import Table
from uncertainties import ufloat, umath

APPOINTMENT = {"Rin": "0.1 1 1 100 100"}


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
        self.extract_data()

        self.xcmfile = None

    @staticmethod
    def from_xcmfile(xcmfile, prexcmfile=None):
        """
        从XCM文件中生成log，返回LogDataResolver对象
        """

        logfile = xcmfile.replace(".xcm", ".log")
        nowpath = os.getcwd()
        target_path = os.path.dirname(xcmfile)
        if target_path:
            os.chdir(target_path)

        def _gen_log(xcmfile, logfile, prexcmfile):
            import xspec

            xspec.AllModels.setEnergies("0.0001 1024. 3500 log")
            if bool(prexcmfile):
                xspec.Xset.restore(prexcmfile)
            xspec.Xset.restore(xcmfile)
            xspec.Xset.openLog(logfile)
            xspec.AllData.show()
            xspec.AllModels.show()
            xspec.Fit.show()
            xspec.Xset.closeLog()

        process = Process(target=_gen_log, args=(xcmfile, logfile, prexcmfile))
        process.start()
        process.join()
        if target_path:
            os.chdir(nowpath)

        loger = LogDataResolver(logfile)
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

        # 模型提取
        model_match = re.search(model_pattern, log_data)
        self.model_used = (
            model_match.group(1).strip() if model_match else "No model found"
        )

        # 统计数据组数
        self.data_groups = len(set(re.findall(r"Data group: (\d+)", log_data)))

        # 模型列表提取
        self.components = list(re.findall(r"\b(\w+)<\d+>", self.model_used))

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

    def stat_spec_result(self, mcmcfile=None, errange=90):
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

        if not bool(mcmcfile):
            return pd.DataFrame.from_dict(free_dict, orient="index", columns=["value"])
        else:
            mcmcchain = open_chain(mcmcfile, modlabel=False)

        efree_dict = {}
        for pname, pvalue in free_dict.items():
            chains = mcmcchain.loc[:, pname]
            percentileleft, percentileright = [
                (100 - errange) / 2,
                100 - (100 - errange) / 2,
            ]
            vL, vU = np.percentile(chains, [percentileleft, percentileright])
            errL, errU = (float(pvalue[0]) - vL), (vU - float(pvalue[0]))
            efree_dict[pname] = [float(pvalue[0]), errL, errU]

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
