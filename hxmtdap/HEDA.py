from datetime import datetime
import logging
import panel as pn
import multiprocessing
import gc

from matplotlib import pyplot as plt

from .core.command import HXMTCommande
from .core.execute import CommandExecutor
from .core.logger import capture_exception_fromM, setup_logger
from .core.parameters import HEParameters
from .core.recode import HERecoder
from .core.status import HEStatus

from .tools.evtutils import plotpds_from_evt
from .tools.lcutils import plotlc, plotbkg
from .tools.pdsutils import plotpds


class HEService(object):
    """
    用于管理数据处理中各模块实例的类
    """

    def __new__(cls, exppath, *args, **kwargs):
        return super(HEService, cls).__new__(cls)

    def __init__(self, exppath) -> None:
        self.initialize(exppath)

    def initialize(self, exppath):
        self.exppath = exppath
        self.logger = setup_logger("HXMTDAS_HE", log_directory=f"{exppath}/log")
        self.check_exp_path(exppath)

        self.recoder = HERecoder(
            logger=self.logger.getChild("Recoder"),
            exppath=exppath,
        )
        self.status = HEStatus(self.recoder, self.logger.getChild("Status"))
        self.Parameters = HEParameters(
            self.recoder, exppath, self.logger.getChild("Parameters")
        )
        self.runner = CommandExecutor(self.logger.getChild("Executor"))
        self.commander = HXMTCommande()
        self._is_initializied = True
        self.logger.debug(f"Initialized {type(self).__name__}.")

    @capture_exception_fromM
    def check_exp_path(self, path):
        # 检测工作目录有效性，放在方法中实现是为了捕捉异常
        HEStatus.check_exp_path(path)

    def get_logger(self):
        return self.logger

    def get_recoder(self):
        return self.recoder

    def get_status(self):
        return self.status

    def get_parameters(self):
        return self.Parameters

    def get_runner(self):
        return self.runner

    def get_commander(self):
        return self.commander

    def get_exppath(self):
        return self.exppath


class HEBasePipeline(object):

    def __init__(self, exppath, logger_level):
        self.Service = HEService(exppath)  # Service实例
        self.Service.initialize(exppath)

        self.exppath = exppath

        self.logger = self.Service.get_logger()
        self.logger_level = logger_level
        self.logger.setLevel(logger_level)

        self.recoder = self.Service.get_recoder()
        self.status = self.Service.get_status()
        self.Parameters = self.Service.get_parameters()
        self.runner = self.Service.get_runner()
        self.commander = self.Service.get_commander()

        self.fig_processes = []

    def save_figure(self, output, plot_func):
        try:
            fig, _ = plot_func(output)
            fig.savefig(f"{output}.png")
            plt.clf()
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to save figure: {e}")


class HEScreenPipeline(HEBasePipeline):

    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    @capture_exception_fromM
    def _initialization(self):
        self.logger.debug("LEScreenPipeline Initialized.")

    def show(self):
        figure = self.recoder.show()
        pn.panel(figure).show()  # 前端显示figure

    @capture_exception_fromM
    def hepical(self, **kwargs):
        self.recoder.get_1L()
        defParams = self.Parameters.hepical(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.hepical(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "2_calib",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.status.check_same_output(**params)  # 检查输出文件是否已经存在
        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files("HE_pi", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("pi", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def hegtigen(self, **kwargs):
        available = self.status.determine_available_commands("hegtigen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.hegtigen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.hegtigen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "3_screen",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files("HE_gti", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("gti", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def hescreen(self, **kwargs):
        available = self.status.determine_available_commands("hescreen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.hescreen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.hescreen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "3_screen",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files("HE_screen", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("screen", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotpds_from_evt)
            )
            self.fig_processes.append(process)
            process.start()
            return output


class HELightcurvePipeline(HEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        self.logger.info("HE Lightcurve Pipeline is ready.")

    @capture_exception_fromM
    def helcgen(self, node="HE_lcraw", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("helcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 这里extinfo包括能段和时间分辨信息，写入node后用于生成背景时调用
        defParams, extinfo = self.Parameters.helcgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.helcgen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            **extinfo,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("lcraw", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotlc)
            )
            self.fig_processes.append(process)
            process.start()
            return output

    @capture_exception_fromM
    def hebkgmap_lc(self, node="HE_lcbkg", **kwargs):
        available = self.status.determine_available_ext("hebkgmap_lc")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.hebkgmap_lc(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.hebkgmap(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents, ignore_missing=True)
            self.status.update("lcbkg", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotbkg)
            )
            self.fig_processes.append(process)
            process.start()
            return output

    @capture_exception_fromM
    def henetlcgen(self, node="HE_lcnet", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_lctask("henetlcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams, extinfo = self.Parameters.henetlcgen(
            **kwargs
        )  # 获取默认参数和额外信息

        # 这里直接生成了netlc
        params, parents = self.commander.henetlcgen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            **extinfo,
            "layer": "5_prod",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            self.logger.debug(f"Generated {output}.")
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("lcnet", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotlc)
            )
            self.fig_processes.append(process)
            process.start()
            return output

    @capture_exception_fromM
    def henetpdsgen(self, node="HE_pdsnet", **kwargs):
        """
        需要传入必要参数：
        - segment 每段功率谱长度
        - rebin 重分箱

        可选参数：
        - outfile 输出文件名；如果未指定将根据规则生成
        - subtracted_white_noise 是否去除白噪声(默认否False)

        leahy归一
        """

        # 允许直接传入 minE和maxE
        available = self.status.determine_available_lctask("henetpdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.henetpdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsnet
        params, parents = self.commander.henetpdsgen(
            **defParams, logger=self.logger
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "5_prod",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            self.logger.debug(f"Generated {output}.")
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("pdsnet", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotpds)
            )
            self.fig_processes.append(process)
            process.start()
            return output

    @capture_exception_fromM
    def hermspdsgen(self, node="HE_pdsrms", **kwargs):
        """
        需要传入必要参数：
        - segment 每段功率谱长度
        - rebin 重分箱

        可选参数：
        - outfile 输出文件名；如果未指定将根据规则生成
        - subtracted_white_noise 是否去除白噪声(默认是True)

        RMS归一
        """

        # 允许直接传入 minE和maxE
        available = self.status.determine_available_lctask("hermspdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.hermspdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsrms
        params, parents = self.commander.hermspdsgen(
            **defParams, logger=self.logger
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "5_prod",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            self.logger.debug(f"Generated {output}.")
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("pdsrms", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            process = multiprocessing.Process(
                target=self.save_figure, args=(output, plotpds)
            )
            self.fig_processes.append(process)
            process.start()
            return output


class HESpectrumPipeline(HEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        pass

    def hespecgen(self, node="HE_spec", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("hespecgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams, extinfo = self.Parameters.hespecgen(
            **kwargs
        )  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.hespecgen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            **extinfo,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("spec", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    def herspgen(self, node="HE_rsp", **kwargs):
        available = self.status.determine_available_ext("herspgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.herspgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.herspgen(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("rsp", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def hebkgmap_spec(self, node="HE_specbkg", **kwargs):
        available = self.status.determine_available_ext("hebkgmap_spec")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.hebkgmap_spec(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.hebkgmap(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(
                fnode, parents, ignore_missing=True
            )  # 如果使用了临时GTI，则忽略GTI的edge
            self.status.update("specbkg", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    def grppha_he(self, node="HE_grp", **kwargs):
        available = self.status.determine_available_ext("grppha_he")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.grppha_he(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.grppha(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "4_ext",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files(node, file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("grp", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output


class HEDA(HEScreenPipeline, HELightcurvePipeline, HESpectrumPipeline):
    def __init__(self, exppath, logger_level="INFO"):
        HEBasePipeline.__init__(self, exppath, logger_level)
        self.logger.info(f"HEDA initialized.")

    def clean(self):
        self.logger.info(f"HEDA closed.")

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        loggername = self.logger.name
        logging.Logger.manager.loggerDict.pop(loggername, None)

        self._is_closed = True
        del self.logger

        for p in self.fig_processes:
            p.join()
        gc.collect()
