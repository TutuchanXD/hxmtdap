import os
import sys
import logging
import subprocess

from rich.console import Console
from rich.logging import RichHandler


class CommandExecutor:
    """
    执行命令行的类
    """

    def __init__(self, logger=None):
        self.console = Console()
        self.logger = logger if bool(logger) else self._set_logger()

        # 根据环境判断是否设置特定的环境变量
        self.environment = self.check_environment()
        if self.environment == "Jupyter":
            self.setup_environment_for_heasoft()

    def _set_logger(self):
        logger = logging.getLogger("HXMTExecutor")
        logger.setLevel(logging.DEBUG)
        handler = RichHandler(rich_tracebacks=True)
        logger.addHandler(handler)
        return logger

    def check_environment(self):
        """
        检查当前是否在Jupyter或其他环境中运行
        """
        try:
            from IPython import get_ipython

            if "IPKernelApp" in get_ipython().config:
                self.logger.debug("Running in Jupyter...")
                return "Jupyter"
        except Exception:
            return "Non-interactive"

        if sys.stdin.isatty():
            return "Terminal"
        else:
            return "Non-interactive"

    def setup_environment_for_heasoft(self):
        """
        设置HEASOFT所需的环境变量，适用于Jupyter或其他非终端环境
        """
        self.old_noquery = os.environ.get("HEADASNOQUERY")
        self.old_prompt = os.environ.get("HEADASPROMPT")
        os.environ["HEADASNOQUERY"] = ""
        os.environ["HEADASPROMPT"] = "/dev/null"

    def restore_environment(self):
        """
        还原修改过的环境变量
        """
        if hasattr(self, "old_noquery"):
            if self.old_noquery is None:
                del os.environ["HEADASNOQUERY"]
            else:
                os.environ["HEADASNOQUERY"] = self.old_noquery

        if hasattr(self, "old_prompt"):
            if self.old_prompt is None:
                del os.environ["HEADASPROMPT"]
            else:
                os.environ["HEADASPROMPT"] = self.old_prompt

    def execute_command(self, cmd_string):
        """
        执行指定的命令，并在执行前后处理环境变量
        """
        software = cmd_string.split()[0]
        ignore_output_list = [
            "flx2xsp",
        ]  # flx2xsp会输出大量空格
        ignore_exception_list = [
            "hebkgmap",
        ]  # hebkgmap使用了过时方法
        Warning_keywords = [
            "WARNING",
            "Warning",
        ]
        Error_keywords = ["ERROR", "Error", "abandon"]

        try:
            with self.console.status(
                f"[bold green]Executing {software}...", spinner="dots"
            ):
                results = subprocess.run(
                    cmd_string,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                )
                if results.stdout and software not in ignore_output_list:
                    self.logger.debug(results.stdout)
                if results.stderr and software not in ignore_exception_list:
                    if "WARNING" in results.stderr or "Warning" in results.stderr:
                        self.logger.warning(results.stderr)
                    elif (
                        "ERROR" in results.stderr
                        or "Error" in results.stderr
                        or "abandon" in results.stderr
                    ):
                        self.logger.error(results.stderr)
                        raise Exception(results.stderr)
                    else:
                        self.logger.error(results.stderr)
                        raise Exception(results.stderr)
        finally:
            if self.environment == "Jupyter":
                self.restore_environment()

    def run(self, cmd_string):
        """
        别名函数
        """
        self.execute_command(cmd_string)


def gen_cmd_string(command: str, params, ktype: str = "keyword") -> str:
    """
    生成命令字符串
    """
    if not command:
        raise ValueError("The command cannot be empty!")

    if ktype not in ["keyword", "stack", "unix"]:
        raise ValueError(f"{ktype} is an unknown type!")

    if ktype == "keyword":
        if not isinstance(params, dict):
            raise ValueError('Params must be a dictionary when ktype is "keyword"!')
        return f"{command} {format_keyword_params(params)}"

    if ktype == "stack":
        if not isinstance(params, list):
            raise ValueError('Params must be a list when ktype is "stack"!')
        return f"{command} {format_stack_params(params)}"

    if ktype == "unix":
        if not isinstance(params, dict):
            raise ValueError('Params must be a dictionary when ktype is "unix"!')
        return f"{command} {format_unix_params(params)}"


def format_keyword_params(params: dict) -> str:
    """
    格式化命令字符串参数，类型为关键字参数
    """
    params_string = []
    for key, value in params.items():
        if isinstance(value, int) and value >= 0:
            params_string.append(f"{key}={value}")
        elif value in {"yes", "no", "NONE"}:
            params_string.append(f"{key}={value}")
        else:
            params_string.append(f'{key}="{value}"')
    return " ".join(params_string)


def format_stack_params(params: list) -> str:
    """
    格式化命令字符串参数，类型为位置参数
    """
    return " ".join([str(param) for param in params])


def format_unix_params(params: dict) -> str:
    """
    格式化命令字符串参数，类型为unix参数
    """
    params_string = []
    for key, value in params.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        params_string.append(f"--{key}={value}")
    return " ".join(params_string)
