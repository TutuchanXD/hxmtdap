# 用于设置日志系统，包括日志级别、格式和输出位置（控制台和文件）。

import os
import sys
import uuid
import logging
import traceback
from io import StringIO
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from functools import wraps

from rich.logging import RichHandler


def setup_logger(
    logname,
    log_directory="log",
    log_level=logging.INFO,
    console=True,
    console_level=logging.INFO,
    unique_logname=True,
):
    """
    设置日志系统，包括日志级别、格式和输出位置（控制台和文件）。

    - logname: 日志文件名
    - log_directory: 存放日志文件的目录，默认为当前目录下的"log"文件夹
    - log_level: 日志文件级别, 默认为INFO
    - console: 是否输出到控制台
    - console_level: 控制台日志级别
    - unique_logname: 是否使用唯一的日志名
    """

    # 确保日志目录存在
    if log_directory and (not os.path.exists(log_directory)):
        os.makedirs(log_directory, exist_ok=True)

    # 日志文件名
    if logname.endswith(".log"):
        log_file = os.path.join(log_directory, f"{logname}")
    else:
        log_file = os.path.join(log_directory, f"{logname}.log")

    # 如果旧的日志文件存在，重命名它
    # if os.path.exists(log_file):
    #     # 获取文件的最后修改时间
    #     modification_time = os.path.getmtime(log_file)
    #     timestamp = datetime.fromtimestamp(modification_time).strftime("%Y%m%d_%H%M%S")
    #     past_log_file = log_file.replace(".log", f"_{timestamp}.log")
    #     os.rename(log_file, past_log_file)

    # 日志格式
    log_format = "%(asctime)s-%(name)s-%(levelname)s: %(message)s\n"
    datefmt = r"%y-%m-%d %H:%M:%S"

    # 创建一个处理器
    if unique_logname:
        # 如果使用唯一的处理器名
        logger_name = f"{logname}_{str(uuid.uuid4())[:4]}"
    else:
        logger_name = logname
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 如果logging的logger字典中已有同名记录器，先清除其处理器，避免处理器累积
    for handler in logger.handlers[:]:  # 复制列表以在迭代中修改
        logger.removeHandler(handler)

    # 创建用于输出到文件的记录器，每次运行程序轮转
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=60
    )
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
    # file_handler.setLevel(log_level) #不设置handle的日志级别，其将灵活动态继承logger的日志级别
    logger.addHandler(file_handler)

    # 创建用于输出到控制台的记录器
    if console:
        console_handler = RichHandler(rich_tracebacks=True)
        if console_level != log_level:
            console_handler.setLevel(console_level)
        logger.addHandler(console_handler)

    logger.debug(f'Logger "{logname}" initialized.')
    return logger


def find_error_in_logs(*logfname, start_path=None):
    """
    在指定路径及其子目录中查找名为 <logname> 文件中的 'ERROR' 关键字。

    - start_path: 搜索的起始路径，默认为当前工作路径。

    返回值:
    - 包含 'ERROR' 关键字的文件路径列表。
    """

    # 设置起始路径
    if not start_path:
        start_path = os.getcwd()
    elif not os.path.exists(start_path):
        raise Exception("Non-existing start path!")

    error_files = []

    # 遍历起始路径下的目录和文件
    for dpath, dname_lst, fname_lst in os.walk(start_path):
        for fname in fname_lst:
            if fname in logfname:
                file_path = os.path.join(dpath, fname)
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        if "ERROR" in line:
                            error_files.append(file_path)
                            break

    return error_files


def capture_exception(logger):
    """
    装饰器函数，将异常输出重定向到指定的 logger 的 error 日志级别

    示例 装饰器用法:
    @capture_exception(logger)
    def func(*args, **kwargs):
        ...

    示例 直接赋值用法：
    func = capture_exception(logger)(func)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb_str = traceback.format_exception(
                    e, value=e, tb=e.__traceback__, chain=True
                )
                logger.error(
                    f"An exception occurred in {func.__name__}: {''.join(tb_str)}",
                )
                raise RuntimeError(f'Error in "{func.__name__}"') from e

        return wrapper

    return decorator


def capture_exception_fromM(func):
    """
    装饰器函数，将异常输出重定向到指定的 logger 的 error 日志级别

    仅适用于指定了self.logger的类方法
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exception(
                e, value=e, tb=e.__traceback__, chain=True
            )
            self.logger.error(
                f"An exception occurred in {func.__name__}: {''.join(tb_str)}",
            )

            # sys.exit(1) # 这里选择直接终止程序
            raise RuntimeError(f'Error in "{func.__name__}"') from e

    return wrapper


def capture_output(logger, level="info", line_by_line: bool = False):
    """
    装饰器函数，将 print 输出重定向到指定的 logger 的指定日志级别

    示例 装饰器用法:
    @capture_output(logger)
    def func(*args, **kwargs):
        ...

    示例 直接赋值用法：
    func = capture_output(logger)(func)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 保存当前的 stdout
            original_stdout = sys.stdout
            # 创建一个字符串IO对象来捕获输出
            sys.stdout = StringIO()
            try:
                # 执行函数
                result = func(*args, **kwargs)
            finally:
                # 这里放到finally中，防止func以任何形式退出
                print(f"above infomations captured by {logger}")
                sys.stdout.seek(0)
                output = sys.stdout.read()
                sys.stdout = original_stdout
                if output:
                    # 对每行输出进行记录
                    log_level = getattr(logging, level.upper(), logging.INFO)
                    wrapper.log_level = level.upper()
                    if line_by_line:
                        for line in output.strip().split("\n"):
                            logger.log(log_level, line)
                    else:
                        logger.log(log_level, output.strip())
            return result

        return wrapper

    return decorator
