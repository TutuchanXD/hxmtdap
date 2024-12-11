from datetime import datetime
import panel as pn
from typing import Literal
import logging

from matplotlib import pyplot as plt

from .core.command import HXMTCommande
from .core.execute import CommandExecutor
from .core.logger import capture_exception_fromM, setup_logger
from .core.parameters import LEParameters
from .core.recode import LERecoder
from .core.status import LEStatus

from .tools.evtutils import plotpds_from_evt
from .tools.lcutils import plotlc, plotbkg
from .tools.pdsutils import plotpds


class LEService(object):
    """
    用于管理数据处理中各模块实例的类
    """

    def __new__(cls, exppath, *args, **kwargs):
        return super(LEService, cls).__new__(cls)

    def __init__(self, exppath) -> None:
        self.initialize(exppath)

    def initialize(self, exppath):
        self.exppath = exppath
        self.logger = setup_logger("HXMTDAS_LE", log_directory=f"{exppath}/log")
        self.check_exp_path(exppath)

        self.recoder = LERecoder(
            logger=self.logger.getChild("Recoder"),
            exppath=exppath,
        )
        self.status = LEStatus(self.recoder, self.logger.getChild("Status"))
        self.Parameters = LEParameters(
            self.recoder, exppath, self.logger.getChild("Parameters")
        )
        self.runner = CommandExecutor(self.logger.getChild("Executor"))
        self.commander = HXMTCommande()
        self._is_initializied = True
        self.logger.debug(f"Initialized {type(self).__name__}.")

    @capture_exception_fromM
    def check_exp_path(self, path):
        # 检测工作目录有效性，放在方法中实现是为了捕捉异常
        LEStatus.check_exp_path(path)

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


class LEBasePipeline(object):

    def __init__(self, exppath, logger_level):
        self.Service = LEService(
            exppath,
        )  # Service实例
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


class LEScreenPipeline(LEBasePipeline):

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
    def lepical(self, clobber=False, **kwargs):
        self.recoder.get_1L()
        defParams = self.Parameters.lepical(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.lepical(
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
            fnode = self.recoder.add_files("LE_pi", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("pi", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def lerecon(self, **kwargs):
        available = self.status.determine_available_commands("lerecon")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.lerecon(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.lerecon(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "2_calib",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files("LE_recon", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("recon", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def legtigen(self, **kwargs):
        available = self.status.determine_available_commands("legtigen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.legtigen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.legtigen(
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
            fnode = self.recoder.add_files("LE_gtitmp", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("gtitmp", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def legticorr(self, mode: Literal["import", "command"] = "import", **kwargs):
        available = self.status.determine_available_commands("legticorr")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.legticorr(mode=mode, **kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.legticorr(
            **defParams, logger=self.logger
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "3_screen",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        if mode == "command":
            self.logger.debug(f"run {cmd_string}")  # 打印命令行
            self.runner.run(cmd_string)  # 开始执行命令

        output = self.status.check_output(
            params["newgti"]
        )  # 检查输出文件，返回输出文件名
        if output:  # 如果有输出文件
            fnode = self.recoder.add_files("LE_gti", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("gti", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def lescreen(self, **kwargs):
        available = self.status.determine_available_commands("lescreen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.lescreen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.lescreen(
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
            fnode = self.recoder.add_files("LE_screen", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("screen", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")

            # 保存图像
            fig, _ = plotpds_from_evt(output)
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output


class LELightcurvePipeline(LEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        self.logger.info("LE Lightcurve Pipeline is ready.")

    @capture_exception_fromM
    def lelcgen(self, node="LE_lcraw", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("lelcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 这里extinfo包括能段和时间分辨信息，写入node后用于生成背景时调用
        defParams, extinfo = self.Parameters.lelcgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.lelcgen(
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
            fig, _ = plotlc(output)
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output

    @capture_exception_fromM
    def lebkgmap_lc(self, node="LE_lcbkg", **kwargs):
        available = self.status.determine_available_ext("lebkgmap_lc")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.lebkgmap_lc(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.lebkgmap(
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
            fig, _ = plotbkg(output)
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output

    @capture_exception_fromM
    def lenetlcgen(self, node="LE_lcnet", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_lctask("lenetlcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams, extinfo = self.Parameters.lenetlcgen(
            **kwargs
        )  # 获取默认参数和额外信息

        # 这里直接生成了netlc
        params, parents = self.commander.lenetlcgen(
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
            fig, _ = plotlc(output)
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output

    @capture_exception_fromM
    def lenetpdsgen(self, node="LE_pdsnet", **kwargs):
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
        available = self.status.determine_available_lctask("lenetpdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.lenetpdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsnet
        params, parents = self.commander.lenetpdsgen(
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
            fig, _ = plotpds(output)
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output

    @capture_exception_fromM
    def lermspdsgen(self, node="LE_pdsrms", **kwargs):
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
        available = self.status.determine_available_lctask("lermspdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.lermspdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsrms
        params, parents = self.commander.lermspdsgen(
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
            fig, _ = plotpds(output)
            try:
                fig.savefig(f"{output}.png")
                plt.close(fig)
            except OverflowError:
                self.logger.warning(f"OverflowError when saving {output}.png")
            del fig, _

            return output


class LESpectrumPipeline(LEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        pass

    def lespecgen(self, node="LE_spec", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("lespecgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams, extinfo = self.Parameters.lespecgen(
            **kwargs
        )  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.lespecgen(
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

    def lerspgen(self, node="LE_rsp", **kwargs):
        available = self.status.determine_available_ext("lerspgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.lerspgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.lerspgen(
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
    def lebkgmap_spec(self, node="LE_specbkg", **kwargs):
        available = self.status.determine_available_ext("lebkgmap_spec")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.lebkgmap_spec(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.lebkgmap(
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

    def grppha_le(self, node="LE_grp", **kwargs):
        available = self.status.determine_available_ext("grppha_le")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.grppha_le(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.grppha(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "5_prod",  # 将参数打包为字典
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


class LEDA(LEScreenPipeline, LELightcurvePipeline, LESpectrumPipeline):
    def __init__(self, exppath, logger_level="INFO"):
        LEBasePipeline.__init__(self, exppath, logger_level)
        self.logger.info(f"LEDA initialized.")
        self._is_closed = False

    def clean(self):
        self.logger.info(f"LEDA closed.")

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        loggername = self.logger.name
        logging.Logger.manager.loggerDict.pop(loggername, None)

        self._is_closed = True
        del self.logger
