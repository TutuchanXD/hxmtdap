# 这个模块的目的是为了实现追踪和管理，用于判断当前状态是否适合进行下一个步骤
import os
from datetime import datetime
from pathlib import Path
from glob import glob
from abc import ABC, abstractmethod

from .logger import capture_exception_fromM
from ..tools.utils import get_expID


class BaseStatus(ABC):
    def __init__(self, recoder, logger):
        self.recoder = recoder
        self.G = recoder.FileGraph
        self.logger = logger

        if not self.G.has_node("STATUS"):
            attrs = {
                "layer": "STATUS",
                "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
            }
            self.G.add_node("STATUS", **attrs)
            self.logger.debug(f"STATUS node created.")
        else:
            self.logger.debug(f"STATUS node exists.")
        self.status = self.G.nodes["STATUS"]  # 图状态节点
        self.p = Path(self.recoder.exppath)  # 曝光号目录
        self.recoder.set_status()  # 为Recoder类增加一个属性：Recoder.status

    @abstractmethod
    def update_status(self, label, node_name):
        pass

    def update(self, label, node_name):
        self.update_status(label, node_name)

    @staticmethod
    def check_exp_path(path):
        _, whether_basename = get_expID(path, check_basename=True)
        if whether_basename:
            pass
        else:
            raise ValueError(f"{path} is not a basename.")

    @capture_exception_fromM
    def check_same_output(self, **params):
        # 检查即将运行的程序的输出文件是否已经存在
        try:
            outfile = params["outfile"]
        except KeyError:
            raise ValueError("Needs an 'outfile' key.")

        matching_files = glob(f"{outfile}*")
        if matching_files:
            self.logger.info(f"{outfile} already exists. Overwriting it.")
        else:
            pass

    @capture_exception_fromM
    def check_output(self, outfile):
        file_extensions = ["lc", "fps", "fits", "pha", "rsp"]
        matching_files = [
            f
            for f in glob(f"{outfile}*")
            if os.path.isfile(f)
            and f.startswith(outfile)
            and f.split(".")[-1] in file_extensions
        ]
        if not matching_files:
            raise FileNotFoundError(f"{outfile} not found.")

        # 按文件名长度排序
        exact_matches = sorted(matching_files, key=len)

        self.logger.debug(f"Matched files: {exact_matches}.")
        return exact_matches[0]

    @capture_exception_fromM
    def gen_ascii_forbkg(self, fnode):
        filename = fnode["file"]
        dirname = os.path.dirname(filename)
        with open(f"{dirname}/forbkg.txt", "w") as f:
            f.write(f"{filename}")
        self.logger.debug(
            f'{filename} writed in {os.path.relpath(f"{dirname}/forbkg.txt", self.p)}'
        )


class LEStatus(BaseStatus):
    def __init__(self, recoder, logger):
        super().__init__(recoder, logger)

        self.labels = {
            "pi": None,
            "recon": None,
            "gtitmp": None,
            "gti": None,
            "screen": None,
            "lcraw": None,
            "lcbkg": None,
            "lcnet": None,
            "pdsnet": None,
            "pdsrms": None,
            "spec": None,
            "rsp": None,
            "specbkg": None,
            "grp": None,
        }
        # 如果已加载图的STATUS节点中不包含状态属性，则添加
        for label, state in self.labels.items():
            if label not in self.status:
                self.status.update({label: state})
                self.logger.debug(f"STATUS node updated label: {label}: {state}.")
        self.logger.debug(f"Status initialized.")

    @capture_exception_fromM
    def update_status(self, label, node):
        try:
            self.status[label] = node
            self.logger.debug(f"Status {label} updated as {node}.")
        except KeyError:
            raise KeyError(f"Invalid command {label}.")

    @capture_exception_fromM
    def determine_available_commands(self, stage):
        # 根据当前阶段确定可用的命令
        if stage == "lepical":
            return True
        elif stage == "lerecon":
            return bool(self.status["pi"])
        elif stage == "legtigen":
            return True
        elif stage == "legticorr":
            return bool(self.status["recon"]) and bool(self.status["gtitmp"])
        elif stage == "lescreen":
            return bool(self.status["recon"]) and bool(self.status["gti"])
        else:
            raise Exception(f"Unknown stage: {stage}")

    @capture_exception_fromM
    def determine_available_ext(self, stage):
        if stage == "lelcgen":
            return bool(self.status["screen"])
        elif stage == "lebkgmap_lc":
            return bool(self.status["lcraw"]) and bool(self.status["gti"])
        elif stage == "lespecgen":
            return bool(self.status["screen"])
        elif stage == "lerspgen":
            return bool(self.status["spec"])
        elif stage == "lebkgmap_spec":
            return bool(self.status["spec"]) and bool(self.status["gti"])
        elif stage == "grppha_le":
            return (
                bool(self.status["spec"])
                and bool(self.status["rsp"])
                and bool(self.status["specbkg"])
            )
        else:
            raise Exception(f"Unknown stage: {stage}")

    def determine_available_lctask(self, stage):
        if stage == "lenetlcgen":
            return bool(self.status["lcraw"]) and bool(self.status["lcbkg"])
        elif stage == "lenetpdsgen":
            return bool(self.status["lcnet"])
        elif stage == "lermspdsgen":
            return bool(self.status["pdsnet"])
        else:
            raise Exception(f"Unknown stage: {stage}")


class MEStatus(BaseStatus):
    def __init__(self, recoder, logger):
        super().__init__(recoder, logger)

        self.labels = {
            "pi": None,
            "grade": None,
            "deadf": None,
            "gtitmp": None,
            "gti": None,
            "baddet": None,
            "screen": None,
            "lcraw": None,
            "lcbkg": None,
            "lcnet": None,
            "pdsnet": None,
            "pdsrms": None,
            "spec": None,
            "rsp": None,
            "specbkg": None,
            "grp": None,
        }
        # 如果已加载图的STATUS节点中不包含状态属性，则添加
        for label, state in self.labels.items():
            if label not in self.status:
                self.status.update({label: state})
                self.logger.debug(f"STATUS node updated label: {label}: {state}.")
        self.logger.debug(f"Status initialized.")

    @capture_exception_fromM
    def update_status(self, label, node):
        try:
            self.status[label] = node
            self.logger.debug(f"Status {label} updated as {node}.")
        except KeyError:
            raise KeyError(f"Invalid command {label}.")

    @capture_exception_fromM
    def determine_available_commands(self, stage):
        # 根据当前阶段确定可用的命令
        if stage == "mepical":
            return True
        elif stage == "megrade":
            return bool(self.status["pi"])
        elif stage == "megtigen":
            return True
        elif stage == "megticorr":
            return bool(self.status["grade"]) and bool(self.status["gtitmp"])
        elif stage == "mescreen":
            return (
                bool(self.status["grade"])
                and bool(self.status["baddet"])
                and bool(self.status["gti"])
            )
        else:
            raise Exception(f"Unknown stage: {stage}")

    @capture_exception_fromM
    def determine_available_ext(self, stage):
        if stage == "melcgen":
            return bool(self.status["screen"]) and bool(self.status["deadf"])
        elif stage == "mebkgmap_lc":
            return (
                bool(self.status["lcraw"])
                and bool(self.status["gti"])
                and bool(self.status["deadf"])
                and bool(self.status["baddet"])
            )
        elif stage == "mespecgen":
            return bool(self.status["screen"]) and bool(self.status["baddet"])
        elif stage == "merspgen":
            return bool(self.status["spec"])
        elif stage == "mebkgmap_spec":
            return (
                bool(self.status["screen"])
                and bool(self.status["spec"])
                and bool(self.status["gti"])
                and bool(self.status["baddet"])
            )
        elif stage == "grppha_me":
            return (
                bool(self.status["spec"])
                and bool(self.status["rsp"])
                and bool(self.status["specbkg"])
            )
        else:
            raise Exception(f"Unknown stage: {stage}")

    def determine_available_lctask(self, stage):
        if stage == "menetlcgen":
            return bool(self.status["lcraw"]) and bool(self.status["lcbkg"])
        elif stage == "menetpdsgen":
            return bool(self.status["lcnet"])
        elif stage == "mermspdsgen":
            return bool(self.status["pdsnet"])
        else:
            raise Exception(f"Unknown stage: {stage}")


class HEStatus(BaseStatus):
    def __init__(self, recoder, logger):
        super().__init__(recoder, logger)

        self.labels = {
            "pi": None,
            "gti": None,
            "screen": None,
            "lcraw": None,
            "lcbkg": None,
            "lcnet": None,
            "lcblind": None,
            "pdsnet": None,
            "pdsblind": None,
            "pdsrms": None,
            "spec": None,
            "rsp": None,
            "specbkg": None,
            "grp": None,
        }
        # 如果已加载图的STATUS节点中不包含状态属性，则添加
        for label, state in self.labels.items():
            if label not in self.status:
                self.status.update({label: state})
                self.logger.debug(f"STATUS node updated label: {label}: {state}.")
        self.logger.debug(f"Status initialized.")

    @capture_exception_fromM
    def update_status(self, label, node):
        try:
            self.status[label] = node
            self.logger.debug(f"Status {label} updated as {node}.")
        except KeyError:
            raise KeyError(f"Invalid command {label}.")

    @capture_exception_fromM
    def determine_available_commands(self, stage):
        # 根据当前阶段确定可用的命令
        if stage == "hepical":
            return True
        elif stage == "hegtigen":
            return True
        elif stage == "hescreen":
            return bool(self.status["pi"]) and bool(self.status["gti"])
        else:
            raise Exception(f"Unknown stage: {stage}")

    @capture_exception_fromM
    def determine_available_ext(self, stage):
        if stage == "helcgen":
            return bool(self.status["screen"])
        elif stage == "hebkgmap_lc":
            return bool(self.status["lcraw"]) and bool(self.status["gti"])
        elif stage == "hespecgen":
            return bool(self.status["screen"])
        elif stage == "herspgen":
            return bool(self.status["spec"])
        elif stage == "hebkgmap_spec":
            return (
                bool(self.status["screen"])
                and bool(self.status["spec"])
                and bool(self.status["gti"])
            )
        elif stage == "grppha_he":
            return (
                bool(self.status["spec"])
                and bool(self.status["rsp"])
                and bool(self.status["specbkg"])
            )
        else:
            raise Exception(f"Unknown stage: {stage}")

    def determine_available_lctask(self, stage):
        if stage == "henetlcgen":
            return bool(self.status["lcraw"]) and bool(self.status["lcbkg"])
        elif stage == "henetpdsgen":
            return bool(self.status["lcnet"])
        elif stage == "heblindpdsgen":
            return bool(self.status["lcblind"])
        elif stage == "hermspdsgen":
            return bool(self.status["pdsnet"])
        else:
            raise Exception(f"Unknown stage: {stage}")
