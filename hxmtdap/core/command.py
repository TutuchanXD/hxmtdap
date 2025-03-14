import gc
import inspect
import json
import logging
import shutil
import importlib.machinery
from typing import Literal

from .execute import gen_cmd_string
from .logger import capture_output
from ..tools.lcutils import generate_netlc
from ..tools.pdsutils import generate_pds_from_lc
from ..tools.pdsutils import save_as_file as savepds
from ..tools.pdsutils import fps2xsp


class HXMTCommande:
    """
    HXMTDAS的命令集
    """

    def __init__(self): ...

    @staticmethod
    def lepical(evtfile, tempfile, outfile, clobber="yes", history="yes", **kwards):
        RequiredParameters = {
            "evtfile": evtfile,
            "tempfile": tempfile,
            "outfile": outfile,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("lepical", Parameters), (evtfile, tempfile)

    @staticmethod
    def lerecon(evtfile, outfile, instatusfile, clobber="yes", history="yes", **kwards):
        RequiredParameters = {
            "evtfile": evtfile,  # 这里需要的是PI文件
            "outfile": outfile,
            "instatusfile": instatusfile,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return (
            Parameters,
            gen_cmd_string("lerecon", Parameters),
            (evtfile, instatusfile),
        )

    @staticmethod
    def legtigen(
        evtfile,
        instatusfile,
        tempfile,
        ehkfile,
        outfile,
        defaultexpr="NONE",
        expr="ELV>10&&DYE_ELV>30&&COR>8&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300&&ANG_DIST<=0.04",
        clobber="yes",
        history="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "instatusfile": instatusfile,
            "tempfile": tempfile,
            "ehkfile": ehkfile,
            "outfile": outfile,
            "defaultexpr": defaultexpr,
            "expr": expr,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return (
            Parameters,
            gen_cmd_string("legtigen", Parameters),
            (evtfile, instatusfile, tempfile, ehkfile),
        )

    @staticmethod
    def legticorr(
        reconfile,
        oldgti,
        newgti,
        mode: Literal["import", "command"] = "import",
        sigma: float = 2.5,
        dtime: float = 0.0,
        logger=None,
    ):
        Parameters = {"reconfile": reconfile, "oldgti": oldgti, "newgti": newgti}
        if mode == "command":
            # ^ 返回位置参数

            return (
                Parameters,
                gen_cmd_string("legticorr", list(Parameters.values()), ktype="stack"),
                (reconfile, oldgti),
            )
        elif mode == "import":
            # 返回legticorr命令绝对路径
            legticorr_path = shutil.which("legticorr")
            if not bool(legticorr_path):
                raise Exception(
                    "legticorr not found! Please check whether HXMTDASoftware is fully installed!"
                )

            # 这里指定加载器加载，因为源码文件无扩展名，无法自行选择加载器
            legticorr_loader = importlib.machinery.SourceFileLoader(
                "legticorr", legticorr_path
            )
            legticorr = legticorr_loader.load_module()
            findgti = capture_output(
                logger, level=logging.getLevelName(logger.getEffectiveLevel())
            )(legticorr.findgti)
            try:
                findgti(
                    reconfile,
                    oldgti,
                    newgti,
                    SVer="2.06",
                    BVer="2.0.14",
                    sigma=sigma,
                    dtime=dtime,
                )
            except SystemExit as e:
                logger.error(e)

            Parameters.update({"sigma": sigma, "dtime": dtime})
            return (
                Parameters,
                None,
                (reconfile, oldgti),
            )

    @staticmethod
    def lescreen(
        evtfile,
        gtifile,
        outfile,
        userdetid="0-95",
        starttime=0,
        stoptime=0,
        clobber="yes",
        history="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "gtifile": gtifile,
            "outfile": outfile,
            "userdetid": userdetid,
            "starttime": starttime,
            "stoptime": stoptime,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("lescreen", Parameters), (evtfile, gtifile)

    @staticmethod
    def lelcgen(
        evtfile,
        outfile,
        binsize,
        userdetid="0,2-4,6-10,12,14,20,22-26,28,30,32,34-36,38-42,44,46,52,54-58,60-62,64,66-68,70-74,76,78,84,86,88-90,92-94",
        starttime=0,
        stoptime=0,
        minPI=106,
        maxPI=1169,
        eventtype=1,
        clobber=1,
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,  # 这里只是前缀
            "outfile": outfile,
            "userdetid": userdetid,
            "binsize": binsize,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "eventtype": eventtype,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("lelcgen", Parameters), (evtfile,)

    @staticmethod
    def lenetlcgen(lcraw, lcbkg, outfile):
        RequiredParameters = {"lcraw": lcraw, "lcbkg": lcbkg, "outfile": outfile}
        Parameters = RequiredParameters.copy()

        lcnet_hdul = generate_netlc(lcraw, lcbkg)
        lcnet_hdul.writeto(outfile, overwrite=True)
        return Parameters, (lcraw, lcbkg)

    @staticmethod
    def lenetpdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="leahy",
        subtracted_white_noise=False,
        logger=None,
    ):
        # ^ 先生成功率谱再传递参数列表，因为segment可能会变
        pdsnet_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )
        # segment从功率谱obj中读取
        real_segment = pdsnet_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsnet_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsnet_obj.power, pdsnet_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def lermspdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="rms",
        subtracted_white_noise=True,
        logger=None,
    ):
        # ^ 先生成功率谱再传递参数列表，因为segment可能会变
        pdsrms_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )

        # segment从功率谱obj中读取
        real_segment = pdsrms_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsrms_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsrms_obj.power, pdsrms_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def lespecgen(
        evtfile,
        outfile,
        eventtype=1,
        userdetid="0,2-4,6-10,12,14,20,22-26,28,30,32,34-36,38-42,44,46,52,54-58,60-62,64,66-68,70-74,76,78,84,86,88-90,92-94",
        starttime=0,
        stoptime=0,
        minPI=0,
        maxPI=1535,
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "eventtype": eventtype,
            "userdetid": userdetid,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("lespecgen", Parameters), (evtfile,)

    @staticmethod
    def lerspgen(
        phafile, outfile, attfile, tempfile, ra="-1", dec="-91", clobber="yes", **kwards
    ):
        RequiredParameters = {
            "phafile": phafile,
            "outfile": outfile,
            "attfile": attfile,
            "tempfile": tempfile,
            "ra": ra,
            "dec": dec,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return (
            Parameters,
            gen_cmd_string("lerspgen", Parameters),
            (phafile, attfile, tempfile),
        )

    @staticmethod
    def lebkgmap(
        ftype,
        evtfile,
        gtifile,
        fascii,
        outfile,
        minPI=0,
        maxPI=1535,
    ):
        # ^ 返回位置参数
        RequiredParameters = {
            "ftype": ftype,
            "evtfile": evtfile,
            "gtifile": gtifile,
            "fascii": fascii,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": outfile,
        }
        Parameters = RequiredParameters.copy()

        with open(fascii, "r") as f:
            extfile = f.readline().strip()  # 返回光变曲线或者能谱的文件名

        return (
            Parameters,
            gen_cmd_string("lebkgmap", list(Parameters.values()), ktype="stack"),
            (evtfile, gtifile, extfile),
        )

    @staticmethod
    def mepical(evtfile, tempfile, outfile, clobber="yes", history="yes", **kwards):
        RequiredParameters = {
            "evtfile": evtfile,
            "tempfile": tempfile,
            "outfile": outfile,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("mepical", Parameters), (evtfile, tempfile)

    @staticmethod
    def megrade(
        evtfile, outfile, deadfile, binsize=1, clobber="yes", history="yes", **kwards
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "deadfile": deadfile,
            "binsize": binsize,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("megrade", Parameters), (evtfile,)

    @staticmethod
    def megtigen(
        tempfile, ehkfile, outfile, defaultexpr, expr, clobber, history, **kwards
    ):
        RequiredParameters = {
            "tempfile": tempfile,
            "ehkfile": ehkfile,
            "outfile": outfile,
            "expr": expr,
            "defaultexpr": defaultexpr,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("megtigen", Parameters), (tempfile, ehkfile)

    @staticmethod
    def megticorr(gradefile, oldgti, newgti, detectorstatus, baddetfile):
        # ^ 返回位置参数
        Parameters = {
            "gradefile": gradefile,
            "oldgti": oldgti,
            "newgti": newgti,
            "detectorstatus": detectorstatus,
            "baddetfile": baddetfile,
        }
        return (
            Parameters,
            gen_cmd_string("megticorr", list(Parameters.values()), ktype="stack"),
            (gradefile, oldgti),
        )

    @staticmethod
    def mescreen(
        evtfile,
        gtifile,
        outfile,
        baddetfile,
        userdetid="0-7,11-25,29-43,47-53",
        starttime=0,
        stoptime=0,
        minPI=0,
        maxPI=1023,
        clobber="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "gtifile": gtifile,
            "outfile": outfile,
            "userdetid": userdetid,
            "baddetfile": baddetfile,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return (
            Parameters,
            gen_cmd_string("mescreen", Parameters),
            (evtfile, gtifile, baddetfile),
        )

    @staticmethod
    def melcgen(
        evtfile,
        outfile,
        deadfile,
        userdetid,
        starttime=0,
        stoptime=0,
        minPI="119",
        maxPI="546",
        deadcorr="yes",
        clobber="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "deadfile": deadfile,
            "userdetid": userdetid,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "deadcorr": deadcorr,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(**kwards)
        return Parameters, gen_cmd_string("melcgen", Parameters), (evtfile, deadfile)

    @staticmethod
    def mespecgen(
        evtfile,
        outfile,
        deadfile,
        userdetid="0-7,11-25,29-43,47-53",
        starttime=0,
        stoptime=0,
        minPI=0,
        maxPI=1023,
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "deadfile": deadfile,
            "userdetid": userdetid,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("mespecgen", Parameters), (evtfile, deadfile)

    @staticmethod
    def menetlcgen(lcraw, lcbkg, outfile):
        RequiredParameters = {"lcraw": lcraw, "lcbkg": lcbkg, "outfile": outfile}
        Parameters = RequiredParameters.copy()

        lcnet_hdul = generate_netlc(lcraw, lcbkg)
        lcnet_hdul.writeto(outfile, overwrite=True)
        return Parameters, (lcraw, lcbkg)

    @staticmethod
    def menetpdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="leahy",
        subtracted_white_noise=False,
        logger=None,
    ):
        pdsnet_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )

        # segment从功率谱obj中读取
        real_segment = pdsnet_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsnet_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsnet_obj.power, pdsnet_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def mermspdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="rms",
        subtracted_white_noise=True,
        logger=None,
    ):
        pdsrms_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )

        # segment从功率谱obj中读取
        real_segment = pdsrms_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsrms_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsrms_obj.power, pdsrms_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def merspgen(
        phafile, outfile, attfile, ra="-1", dec="-91", clobber="yes", **kwards
    ):
        RequiredParameters = {
            "phafile": phafile,
            "outfile": outfile,
            "attfile": attfile,
            "ra": ra,
            "dec": dec,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("merspgen", Parameters), (phafile, attfile)

    @staticmethod
    def mebkgmap(
        ftype,
        evtfile,
        ehkfile,
        gtifile,
        dtimefile,
        thfile,
        fascii,
        minPI,
        maxPI,
        outfile,
        statusfile,
    ):
        # ^ 返回位置参数
        RequiredParameters = {
            "ftype": ftype,
            "evtfile": evtfile,
            "ehkfile": ehkfile,
            "gtifile": gtifile,
            "dtimefile": dtimefile,
            "thfile": thfile,
            "fascii": fascii,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": outfile,
            "statusfile": statusfile,
        }
        Parameters = RequiredParameters.copy()

        with open(fascii, "r") as f:
            extfile = f.readline().strip()  # 返回光变曲线或者能谱的文件名

        return (
            Parameters,
            gen_cmd_string("mebkgmap", list(Parameters.values()), ktype="stack"),
            (evtfile, dtimefile, statusfile, extfile),
        )

    @staticmethod
    def hepical(
        evtfile, outfile, minpulsewidth=54, maxpulsewidth=70, clobber="yes", **kwards
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "minpulsewidth": minpulsewidth,
            "maxpulsewidth": maxpulsewidth,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("hepical", Parameters), (evtfile,)

    @staticmethod
    def hegtigen(
        hvfile,
        tempfile,
        pmfile,
        outfile,
        ehkfile,
        defaultexpr="NONE",
        expr="ELV>10&&COR>8&&SAA_FLAG==0&&TN_SAA>300&&T_SAA>300&&ANG_DIST<=0.04",
        pmexpr="",
        clobber="yes",
        history="yes",
        **kwards,
    ):
        RequiredParameters = {
            "hvfile": hvfile,
            "tempfile": tempfile,
            "pmfile": pmfile,
            "outfile": outfile,
            "ehkfile": ehkfile,
            "defaultexpr": defaultexpr,
            "expr": expr,
            "pmexpr": pmexpr,
            "clobber": clobber,
            "history": history,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return (
            Parameters,
            gen_cmd_string("hegtigen", Parameters),
            (hvfile, tempfile, pmfile, ehkfile),
        )

    @staticmethod
    def hescreen(
        evtfile,
        gtifile,
        outfile,
        userdetid="0-17",
        eventtype=1,
        anticoincidence="yes",
        starttime=0,
        stoptime=0,
        minPI=0,
        maxPI=255,
        clobber="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "gtifile": gtifile,
            "outfile": outfile,
            "userdetid": userdetid,
            "eventtype": eventtype,
            "anticoincidence": anticoincidence,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("hescreen", Parameters), (evtfile, gtifile)

    @staticmethod
    def helcgen(
        evtfile,
        outfile,
        deadfile,
        userdetid,
        binsize=1,
        starttime=0,
        stoptime=0,
        minPI=0,
        maxPI=162,
        deadcorr="yes",
        clobber="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "deadfile": deadfile,
            "userdetid": userdetid,
            "binsize": binsize,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "deadcorr": deadcorr,
            "clobber": clobber,
        }

        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("helcgen", Parameters), (evtfile, deadfile)

    @staticmethod
    def henetlcgen(lcraw, lcbkg, outfile):
        RequiredParameters = {"lcraw": lcraw, "lcbkg": lcbkg, "outfile": outfile}
        Parameters = RequiredParameters.copy()

        lcnet_hdul = generate_netlc(lcraw, lcbkg)
        lcnet_hdul.writeto(outfile, overwrite=True)
        return Parameters, (lcraw, lcbkg)

    @staticmethod
    def henetpdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="leahy",
        subtracted_white_noise=False,
        logger=None,
    ):
        pdsnet_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )

        # segment从功率谱obj中读取
        real_segment = pdsnet_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsnet_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsnet_obj.power, pdsnet_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def hermspdsgen(
        lcnet,
        outfile,
        segment=256.0,
        rebin=0,
        norm="rms",
        subtracted_white_noise=True,
        logger=None,
    ):
        pdsrms_obj = generate_pds_from_lc(
            lc=lcnet,
            segment=segment,
            rebin=rebin,
            norm=norm,
            subtracted_white_noise=subtracted_white_noise,
            logger=logger,
        )

        # segment从功率谱obj中读取
        real_segment = pdsrms_obj.segment_size

        RequiredParameters = {
            "lcnet": lcnet,
            "segment": real_segment,
            "rebin": rebin,
            "norm": norm,
            "subtracted_white_noise": subtracted_white_noise,
        }
        Parameters = RequiredParameters.copy()
        outfile = outfile.replace(f"{int(segment)}s", f"{int(real_segment)}s")
        Parameters.update({"outfile": outfile})

        savepds(pdsrms_obj, filename=outfile)
        fps2xsp(fpsfile=outfile, logger=logger)
        del pdsrms_obj.power, pdsrms_obj
        gc.collect()
        return Parameters, (lcnet,)

    @staticmethod
    def hespecgen(
        evtfile,
        outfile,
        deadfile,
        userdetid,
        eventtype=1,
        starttime=0,
        stoptime=0,
        minPI=8,
        maxPI=162,
        clobber="yes",
        **kwards,
    ):
        RequiredParameters = {
            "evtfile": evtfile,
            "outfile": outfile,
            "deadfile": deadfile,
            "userdetid": userdetid,
            "eventtype": eventtype,
            "starttime": starttime,
            "stoptime": stoptime,
            "minPI": minPI,
            "maxPI": maxPI,
            "clobber": clobber,
        }

        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)

        return Parameters, gen_cmd_string("hespecgen", Parameters), (evtfile, deadfile)

    @staticmethod
    def herspgen(
        phafile, outfile, attfile, ra="-1", dec="-91", clobber="yes", **kwards
    ):
        RequiredParameters = {
            "phafile": phafile,
            "outfile": outfile,
            "attfile": attfile,
            "ra": ra,
            "dec": dec,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        Parameters.update(kwards)
        return Parameters, gen_cmd_string("herspgen", Parameters), (phafile, attfile)

    @staticmethod
    def hebkgmap(
        ftype, evtfile, ehkfile, gtifile, dtimefile, fascii, minPI, maxPI, outfile
    ):
        # ^ 返回位置参数
        RequiredParameters = {
            "ftype": ftype,
            "evtfile": evtfile,
            "ehkfile": ehkfile,
            "gtifile": gtifile,
            "dtimefile": dtimefile,
            "fascii": fascii,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": outfile,
        }

        with open(fascii, "r") as f:
            extfile = f.readline().strip()  # 返回光变曲线或者能谱的文件名

        Parameters = RequiredParameters.copy()

        return (
            Parameters,
            gen_cmd_string("hebkgmap", list(Parameters.values()), ktype="stack"),
            (evtfile, ehkfile, dtimefile, extfile),
        )

    @staticmethod
    def grppha(infile, outfile, group, respfile, backfile, clobber):
        RequiredParameters = {
            "infile": infile,
            "outfile": outfile,
            "group": group,
            "respfile": respfile,
            "backfile": backfile,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()
        cmd_string = (
            f"grppha infile='{infile}' outfile='{outfile}' "
            + f"comm='group {group}& chkey respfile {respfile}& chkey backfile {backfile}& exit' clobber={clobber}"
        )

        return Parameters, cmd_string, (infile, respfile, backfile)

    @staticmethod
    def ftgrouppha(
        infile,
        respfile,
        backfile,
        outfile,
        group=None,  # 为了与grppha方法兼容
        grouptype="opt",
        groupscale=None,
        rows="-",
        clobber="yes",
        **kwargs,
    ):
        """
        grouptype 参数说明:
            1. "bmin":
            - 含义: 根据背景文件中每个 bin 的最小计数设置分组。
            - groupscale: 指定每个 bin 的最小计数。
            - 示例: 每个分组在背景文件中至少有 groupscale 个计数。

            2. "constant":
            - 含义: 按固定因子均匀分组。
            - groupscale: 指定分组因子（例如，groupscale=2 表示每两个通道合并为一个 bin）。
            - 示例: 在 xspec 中按固定因子（如 2）分组。

            3. "min":
            - 含义: 根据输入文件中每个 bin 的最小计数设置分组。
            - groupscale: 指定每个 bin 的最小计数。
            - 示例: 每个分组至少有 groupscale 个计数。

            4. "snmin":
            - 含义: 根据背景扣除后的信噪比（S/N）设置分组。
            - groupscale: 指定每个 bin 的最小信噪比。
            - 示例: 每个分组的背景扣除后信噪比至少为 groupscale。

            5. "opt": (默认)
            - 含义: 使用 Kaastra & Bleeker (2016) 提出的最优分组方案。
            - groupscale: 无直接作用（此模式忽略 groupscale）。
            - 注意: 忽略 minchannel 和 maxchannel 设置。
            - 示例: 根据最优算法分组，通常用于高统计数据。

            6. "optmin":
            - 含义: 结合 "opt" 和 "min"，使用 Kaastra & Bleeker 最优分组，同时要求最小计数。
            - groupscale: 指定每个 bin 的最小计数。
            - 示例: 最优分组且每个 bin 至少有 groupscale 个计数。

            7. "optsnmin":
            - 含义: 结合 "opt" 和 "snmin"，使用 Kaastra & Bleeker 最优分组，同时要求最小信噪比。
            - groupscale: 指定每个 bin 的最小信噪比。
            - 示例: 最优分组且每个 bin 的背景扣除后信噪比至少为 groupscale。

            8. "file":
            - 含义: 从模板文件中读取 GROUPING 列。
            - groupscale: 无作用。
            - 额外参数: 需要 templatefile（模板谱文件）。
            - 示例: 将 templatefile 的分组应用到输入文件。
        """
        RequiredParameters = {
            "infile": infile,
            "respfile": respfile,
            "backfile": backfile,
            "outfile": outfile,
            "grouptype": grouptype,
            "groupscale": groupscale,
            "rows": rows,
            "clobber": clobber,
        }
        Parameters = RequiredParameters.copy()

        return (
            Parameters,
            gen_cmd_string("ftgrouppha", Parameters),
            (infile, respfile, backfile),
        )


def params_to_json(cls):
    """
    暂无用途
    """
    methods_dict = {}
    methods = [
        method
        for method in dir(cls)
        if callable(getattr(cls, method)) and not method.startswith("__")
    ]
    for method_name in methods:
        params_dict = {}
        method = getattr(cls, method_name)
        params = inspect.signature(method).parameters
        for param_name, param in params.items():
            if param.default == inspect.Parameter.empty:
                params_dict[param_name] = None
            else:
                params_dict[param_name] = param.default
        methods_dict[method_name] = params_dict
    return json.dumps(methods_dict, indent=4)
