from datetime import datetime
import logging
import panel as pn

from matplotlib import pyplot as plt

from .core.command import HXMTCommande
from .core.execute import CommandExecutor
from .core.logger import capture_exception_fromM, setup_logger
from .core.parameters import MEParameters
from .core.recode import MERecoder
from .core.status import MEStatus

from .tools.lcutils import plotlc, plotbkg
from .tools.pdsutils import plotpds

class MEService(object):
    """
    用于管理数据处理中各模块实例的类
    """

    def __new__(cls, exppath, *args, **kwargs):
        return super(MEService, cls).__new__(cls)

    def __init__(self, exppath) -> None:
        self.initialize(exppath)

    def initialize(self, exppath):
        self.exppath = exppath
        self.logger = setup_logger("HXMTDAS_ME", log_directory=f"{exppath}/log")
        self.check_exp_path(exppath)

        self.recoder = MERecoder(
            logger=self.logger.getChild("Recoder"),
            exppath=exppath,
        )
        self.status = MEStatus(self.recoder, self.logger.getChild("Status"))
        self.Parameters = MEParameters(
            self.recoder, exppath, self.logger.getChild("Parameters")
        )
        self.runner = CommandExecutor(self.logger.getChild("Executor"))
        self.commander = HXMTCommande()
        self._is_initializied = True
        self.logger.debug(f"Initialized {type(self).__name__}.")

    @capture_exception_fromM
    def check_exp_path(self, path):
        # 检测工作目录有效性，放在方法中实现是为了捕捉异常
        MEStatus.check_exp_path(path)

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


class MEBasePipeline(object):

    def __init__(self, exppath, logger_level):
        self.Service = MEService(exppath)  # Service实例
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


class MEScreenPipeline(MEBasePipeline):

    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    @capture_exception_fromM
    def _initialization(self):
        self.logger.debug("ME Screen Pipeline Initialized.")

    def show(self):
        figure = self.recoder.show()
        pn.panel(figure).show()  # 前端显示figure

    @capture_exception_fromM
    def mepical(self, **kwargs):
        self.recoder.get_1L()
        defParams = self.Parameters.mepical(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.mepical(
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
            fnode = self.recoder.add_files("ME_pi", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("pi", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def megrade(self, **kwargs):
        available = self.status.determine_available_commands("megrade")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.megrade(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.megrade(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "2_calib",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output_grade = self.status.check_output(
            params["outfile"]
        )  # 检查输出文件，返回输出文件名
        if output_grade:  # 如果有输出文件
            fnode_grade = self.recoder.add_files("ME_grade", file=output_grade, **attrs)
            self.recoder.add_parents(fnode_grade, parents)
            self.status.update("grade", fnode_grade)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode_grade}.")

        # 第二个输出文件
        output_deadf = self.status.check_output(params["deadfile"])
        if output_deadf:  # 如果有输出文件
            fnode_deadf = self.recoder.add_files("ME_deadf", file=output_deadf, **attrs)
            self.recoder.add_parents(fnode_deadf, parents)
            self.status.update("deadf", fnode_deadf)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode_deadf}.")
            return output_grade, output_deadf

    @capture_exception_fromM
    def megtigen(self, **kwargs):
        available = self.status.determine_available_commands("megtigen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.megtigen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.megtigen(
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
            fnode = self.recoder.add_files("ME_gtitmp", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("gtitmp", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output

    @capture_exception_fromM
    def megticorr(self, **kwargs):
        available = self.status.determine_available_commands("megticorr")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.megticorr(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.megticorr(
            **defParams
        )  # 获取所有参数、命令行执行命令、父血缘数据

        attrs = {
            **params,
            "layer": "3_screen",  # 将参数打包为字典
            "timestamp": datetime.now().strftime(r"%y-%m-%d %H:%M:%S"),
        }

        self.logger.debug(f"run {cmd_string}")  # 打印命令行

        self.runner.run(cmd_string)  # 开始执行命令
        output_newgti = self.status.check_output(
            params["newgti"]
        )  # 检查输出文件，返回输出文件名
        if output_newgti:  # 如果有输出文件
            fnode_newgti = self.recoder.add_files("ME_gti", file=output_newgti, **attrs)
            self.recoder.add_parents(fnode_newgti, parents)
            self.status.update("gti", fnode_newgti)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode_newgti}.")

        # 第二个输出文件
        output_baddet = self.status.check_output(params["baddetfile"])
        if output_baddet:  # 如果有输出文件
            fnode_baddet = self.recoder.add_files(
                "ME_baddet", file=output_baddet, **attrs
            )
            self.recoder.add_parents(fnode_baddet, parents)
            self.status.update("baddet", fnode_baddet)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode_baddet}.")
            return output_newgti, output_baddet

    @capture_exception_fromM
    def mescreen(self, **kwargs):
        available = self.status.determine_available_commands("mescreen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.mescreen(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.mescreen(
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
            fnode = self.recoder.add_files("ME_screen", file=output, **attrs)
            self.recoder.add_parents(fnode, parents)
            self.status.update("screen", fnode)
            self.recoder.save_graph()
            self.logger.info(f"Generated {fnode}.")
            return output


class MELightcurvePipeline(MEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        self.logger.info("ME Lightcurve Pipeline is ready.")

    @capture_exception_fromM
    def melcgen(self, node="ME_lcraw", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("melcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 这里extinfo包括能段和时间分辨信息，写入node后用于生成背景时调用
        defParams, extinfo = self.Parameters.melcgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.melcgen(
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
    def mebkgmap_lc(self, node="ME_lcbkg", **kwargs):
        available = self.status.determine_available_ext("mebkgmap_lc")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.mebkgmap_lc(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.mebkgmap(
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
    def menetlcgen(self, node="ME_lcnet", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_lctask("menetlcgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams, extinfo = self.Parameters.menetlcgen(
            **kwargs
        )  # 获取默认参数和额外信息

        # 这里直接生成了netlc
        params, parents = self.commander.menetlcgen(
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
    def menetpdsgen(self, node="ME_pdsnet", **kwargs):
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
        available = self.status.determine_available_lctask("menetpdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.menetpdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsnet
        params, parents = self.commander.menetpdsgen(
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
    def mermspdsgen(self, node="ME_pdsrms", **kwargs):
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
        available = self.status.determine_available_lctask("mermspdsgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        # ^ 用于生成功率谱时调用
        defParams = self.Parameters.mermspdsgen(**kwargs)  # 获取默认参数和额外信息

        # 这里直接生成了pdsrms
        params, parents = self.commander.mermspdsgen(
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
            fig.savefig(f"{output}.png")
            plt.close(fig)
            return output


class MESpectrumPipeline(MEBasePipeline):
    def __init__(self, exppath, logger_level="INFO"):
        super().__init__(exppath, logger_level)
        self._initialization()
        if logger_level != self.logger_level:
            self.logger.setLevel(logger_level)
            self.logger.info(f"Logger level set to {self.logger_level}")

    def _initialization(self):
        pass

    def mespecgen(self, node="ME_spec", **kwargs):
        # 允许直接传入 minE和maxE
        available = self.status.determine_available_ext("mespecgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams, extinfo = self.Parameters.mespecgen(
            **kwargs
        )  # 获取默认参数和额外信息，extinfo包括能段和GTI
        params, cmd_string, parents = self.commander.mespecgen(
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

    def merspgen(self, node="ME_rsp", **kwargs):
        available = self.status.determine_available_ext("merspgen")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.merspgen(**kwargs)  # 获取默认参数和额外信息
        params, cmd_string, parents = self.commander.merspgen(
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
    def mebkgmap_spec(self, node="ME_specbkg", **kwargs):
        available = self.status.determine_available_ext("mebkgmap_spec")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.mebkgmap_spec(**kwargs)  # 获取默认参数
        params, cmd_string, parents = self.commander.mebkgmap(
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

    def grppha_me(self, node="ME_grp", **kwargs):
        available = self.status.determine_available_ext("grppha_me")
        if not available:
            self.logger.warning(f"The pre-tasks are not completed.")
            return
        defParams = self.Parameters.grppha_me(**kwargs)  # 获取默认参数
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


class MEDA(MEScreenPipeline, MELightcurvePipeline, MESpectrumPipeline):
    def __init__(self, exppath, logger_level="INFO"):
        MEBasePipeline.__init__(self, exppath, logger_level)
        self.logger.info(f"MEDA initialized.")

    def clean(self):
        self.logger.info(f"MEDA closed.")

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        loggername = self.logger.name
        logging.Logger.manager.loggerDict.pop(loggername, None)
        
        self._is_closed = True
        del self.logger
