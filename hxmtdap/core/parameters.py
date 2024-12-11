import os
import json
import tempfile

from os.path import join as pathjoin

from ..tools.utils import energy_to_pi, pi_to_energy, get_expID
from ..tools.lcutils import matching_time_unit
from ..tools.specutils import generate_GTI


def update_parameters(default_parameters, **kwargs):
    """
    更新参数字典中的键值
    """
    for key in kwargs:
        if key in default_parameters:
            default_parameters[key] = kwargs[key]
    return default_parameters


class DefaultName:  # TODO 应该将输出路径和文件名分开
    """
    用来生成默认文件名和输出路径
    """

    def __init__(self, exppath):
        self.p = exppath

        self.calib = pathjoin(exppath, "calib")
        self.screen = pathjoin(exppath, "screen")
        self.spec = pathjoin(exppath, "spec")
        self.lc = pathjoin(exppath, "lc")
        self.pds = pathjoin(exppath, "pds")
        self.expID = get_expID(exppath)

        self.make_dirs()

    def make_dirs(self):
        os.makedirs(self.calib, exist_ok=True)
        os.makedirs(self.screen, exist_ok=True)
        os.makedirs(self.spec, exist_ok=True)

    def make_dirs_lc(self, dti="LE"):
        os.makedirs(pathjoin(self.lc, dti, "raw"), exist_ok=True)
        os.makedirs(pathjoin(self.lc, dti, "bkg"), exist_ok=True)
        os.makedirs(pathjoin(self.lc, dti, "net"), exist_ok=True)

    def make_dirs_pds(self, dti="LE"):
        os.makedirs(pathjoin(self.pds, dti, "leahy"), exist_ok=True)
        os.makedirs(pathjoin(self.pds, dti, "rms"), exist_ok=True)
        if dti == "HE":
            os.makedirs(pathjoin(self.pds, dti, "blind"), exist_ok=True)

    def make_dirs_spec(self, dti="LE"):
        os.makedirs(pathjoin(self.spec, dti), exist_ok=True)

    @property
    def lepical(self):
        outfilename = f"{self.expID}_LE_pi.fits"
        outfile = pathjoin(self.calib, outfilename)
        return {"outfile": outfile}

    @property
    def lerecon(self):
        outfilename = f"{self.expID}_LE_recon.fits"
        outfile = pathjoin(self.calib, outfilename)
        return {"outfile": outfile}

    @property
    def legtigen(self):
        outfilename = f"{self.expID}_LE_gti_tmp.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    @property
    def legticorr(self):
        newgtiname = f"{self.expID}_LE_gti.fits"
        newgti = pathjoin(self.screen, newgtiname)
        return {"newgti": newgti}

    @property
    def lescreen(self):
        outfilename = f"{self.expID}_LE_screen.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    def lelcgen(self, minE, maxE, timedel, lctype="raw"):
        self.make_dirs_lc("LE")
        outfilename_prefix = f"{self.expID}_LE_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "LE", "raw", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def lebkgmap_lc(self, minE, maxE, timedel, lctype="lcbkg"):
        outfilename_prefix = f"{self.expID}_LE_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "LE", "bkg", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def lenetlcgen(self, minE, maxE, timedel):
        self.make_dirs_lc("LE")
        outfilename = f"{self.expID}_LE_net_{minE}-{maxE}keV_{timedel}Bin.lc"
        outfile = pathjoin(self.lc, "LE", "net", outfilename)
        return {"outfile": outfile}

    def lenetpdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("LE")
        outfilename = (
            f"{self.expID}_LE_{int(segment)}s_leahy_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfile = pathjoin(self.pds, "LE", "leahy", f"{rebin}", outfilename)
        return {"outfile": outfile}

    def lermspdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("LE")
        outfilename = (
            f"{self.expID}_LE_{int(segment)}s_rms_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfile = pathjoin(self.pds, "LE", "rms", f"{rebin}", outfilename)
        return {"outfile": outfile}

    def lespecgen(self, minE, maxE):
        self.make_dirs_spec("LE")
        outfilename_prefix = f"{self.expID}_LE_spec_{minE}-{maxE}keV"
        outfile_prefix = pathjoin(self.spec, "LE", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def lerspgen(self, minE, maxE):
        outfilename = f"{self.expID}_LE_rsp_{minE}-{maxE}keV.fits"
        outfile = pathjoin(self.spec, "LE", outfilename)
        return {"outfile": outfile}

    def lebkgmap_spec(self, minE, maxE, suffix=""):
        outfilename_prefix = f"{self.expID}_LE_specbkg_{minE}-{maxE}keV{'' if not suffix else f'_{suffix}'}"
        outfile_prefix = pathjoin(self.spec, "LE", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def grppha_le(self, minE, maxE):
        outfilename = f"{self.expID}_LE_{minE}-{maxE}keV.pha"
        outfile = pathjoin(self.spec, "LE", outfilename)
        return {"outfile": outfile}

    # & FOLLOWING ARE FOR ME PIPELINE

    @property
    def mepical(self):
        outfilename = f"{self.expID}_ME_pi.fits"
        outfile = pathjoin(self.calib, outfilename)
        return {"outfile": outfile}

    @property
    def megrade(self):
        outfilename = f"{self.expID}_ME_grade.fits"
        deadfilename = f"{self.expID}_ME_dtime.fits"
        outfile = pathjoin(self.calib, outfilename)
        deadfile = pathjoin(self.calib, deadfilename)
        return {"outfile": outfile, "deadfile": deadfile}

    @property
    def megtigen(self):
        outfilename = f"{self.expID}_ME_gti_tmp.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    @property
    def megticorr(self):
        newgtiname = f"{self.expID}_ME_gti.fits"
        baddetfilename = f"{self.expID}_ME_status.fits"
        newgti = pathjoin(self.screen, newgtiname)
        baddetfile = pathjoin(self.screen, baddetfilename)
        return {"newgti": newgti, "baddetfile": baddetfile}

    @property
    def mescreen(self):
        outfilename = f"{self.expID}_ME_screen.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    def melcgen(self, minE, maxE, timedel, lctype="lc"):
        self.make_dirs_lc("ME")
        outfilename_prefix = f"{self.expID}_ME_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "ME", "raw", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def mebkgmap_lc(self, minE, maxE, timedel, lctype="lcbkg"):
        outfilename_prefix = f"{self.expID}_ME_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "ME", "bkg", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def menetlcgen(self, minE, maxE, timedel):
        self.make_dirs_lc("ME")
        outfilename = f"{self.expID}_ME_net_{minE}-{maxE}keV_{timedel}Bin.lc"
        outfile = pathjoin(self.lc, "ME", "net", outfilename)
        return {"outfile": outfile}

    def menetpdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("ME")
        outfimename = (
            f"{self.expID}_ME_{int(segment)}s_leahy_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfime = pathjoin(self.pds, "ME", "leahy", f"{rebin}", outfimename)
        return {"outfile": outfime}

    def mermspdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("ME")
        outfimename = (
            f"{self.expID}_ME_{int(segment)}s_rms_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfime = pathjoin(self.pds, "ME", "rms", f"{rebin}", outfimename)
        return {"outfile": outfime}

    def mespecgen(self, minE, maxE):
        self.make_dirs_spec("ME")
        outfilename_prefix = f"{self.expID}_ME_spec_{minE}-{maxE}keV"
        outfile_prefix = pathjoin(self.spec, "ME", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def merspgen(self, minE, maxE):
        outfilename = f"{self.expID}_ME_rsp_{minE}-{maxE}keV.fits"
        outfile = pathjoin(self.spec, "ME", outfilename)
        return {"outfile": outfile}

    def mebkgmap_spec(self, minE, maxE, suffix=""):
        outfilename_prefix = f"{self.expID}_ME_specbkg_{minE}-{maxE}keV{'' if not suffix else f'_{suffix}'}"
        outfile_prefix = pathjoin(self.spec, "ME", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def grppha_me(self, minE, maxE):
        outfilename = f"{self.expID}_ME_{minE}-{maxE}keV.pha"
        outfile = pathjoin(self.spec, "ME", outfilename)
        return {"outfile": outfile}

    # & FOLLOWING ARE FOR HE PIPELINE

    @property
    def hepical(self):
        outfilename = f"{self.expID}_HE_pi.fits"
        outfile = pathjoin(self.calib, outfilename)
        return {"outfile": outfile}

    @property
    def hegtigen(self):
        outfilename = f"{self.expID}_HE_gti.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    @property
    def hescreen(self):
        outfilename = f"{self.expID}_HE_screen.fits"
        outfile = pathjoin(self.screen, outfilename)
        return {"outfile": outfile}

    def helcgen(self, minE, maxE, timedel, lctype="lc"):
        self.make_dirs_lc("HE")
        outfilename_prefix = f"{self.expID}_HE_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "HE", "raw", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def hebkgmap_lc(self, minE, maxE, timedel, lctype="lcbkg"):
        outfilename_prefix = f"{self.expID}_HE_{lctype}_{minE}-{maxE}keV_{timedel}Bin"
        outfile_prefix = pathjoin(self.lc, "HE", "bkg", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def henetlcgen(self, minE, maxE, timedel):
        self.make_dirs_lc("HE")
        outfilename = f"{self.expID}_HE_net_{minE}-{maxE}keV_{timedel}Bin.lc"
        outfile = pathjoin(self.lc, "HE", "net", outfilename)
        return {"outfile": outfile}

    def henetpdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("HE")
        outfimename = (
            f"{self.expID}_HE_{int(segment)}s_leahy_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfime = pathjoin(self.pds, "HE", "leahy", f"{rebin}", outfimename)
        return {"outfile": outfime}

    def heblindpdsgen(self, minE, maxE, segment, timedel):
        self.make_dirs_pds("HE")
        outfilename = (
            f"{self.expID}_HE_{int(segment)}s_blind_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfile = pathjoin(self.pds, "HE", "blind", outfilename)
        return {"outfile": outfile}

    def hermspdsgen(self, minE, maxE, segment, timedel, rebin):
        self.make_dirs_pds("HE")
        outfimename = (
            f"{self.expID}_HE_{int(segment)}s_rms_{minE}-{maxE}keV_{timedel}Bin.fps"
        )
        outfime = pathjoin(self.pds, "HE", "rms", f"{rebin}", outfimename)
        return {"outfile": outfime}

    def hespecgen(self, minE, maxE):
        self.make_dirs_spec("HE")
        outfilename_prefix = f"{self.expID}_HE_spec_{minE}-{maxE}keV"
        outfile_prefix = pathjoin(self.spec, "HE", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def herspgen(self, minE, maxE):
        outfilename = f"{self.expID}_HE_rsp_{minE}-{maxE}keV.fits"
        outfile = pathjoin(self.spec, "HE", outfilename)
        return {"outfile": outfile}

    def hebkgmap_spec(self, minE, maxE, suffix=""):
        outfilename_prefix = f"{self.expID}_HE_specbkg_{minE}-{maxE}keV{'' if not suffix else f'_{suffix}'}"
        outfile_prefix = pathjoin(self.spec, "HE", outfilename_prefix)
        return {"outfile_prefix": outfile_prefix}

    def grppha_he(self, minE, maxE):
        outfilename = f"{self.expID}_HE_{minE}-{maxE}keV.pha"
        outfile = pathjoin(self.spec, "HE", outfilename)
        return {"outfile": outfile}


class BaseParameters:

    def __init__(self, recoder, exppath, logger):
        self.expID = get_expID(exppath)
        self.recoder = recoder
        self.dner = DefaultName(exppath)
        self.exppath = exppath
        self.logger = logger

    def get_full_filepath(self, node_name):
        """
        一般用来从图的status节点中记录的活动node中获取活动node的绝对文件路径
        """
        return getattr(self.recoder, node_name)  # 获取node_name的绝对文件路径

    def get(self, node_name):
        """
        self.get_full_filepath的别名
        """
        return self.get_full_filepath(node_name)

    def outfile_generator(self, dir, filename):
        pass


class LEParameters(BaseParameters):

    def __init__(self, recoder, exppath, logger):
        super().__init__(recoder, exppath, logger)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = pathjoin(dir_path, "..", "json", "le_parameters.json")
        with open(json_path, "r") as file:
            self.default_parameters = json.load(file)

    def lepical(self, **kwargs):
        default_parameters = {
            "evtfile": self.recoder.LE_Evt,
            "outfile": self.dner.lepical["outfile"],
            "tempfile": self.recoder.LE_TH,
            "clobber": self.default_parameters["lepical"]["clobber"],
            "history": self.default_parameters["lepical"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lerecon(self, **kwargs):
        default_parameters = {
            "evtfile": self.get(self.recoder.status["pi"]),
            "outfile": self.dner.lerecon["outfile"],
            "instatusfile": self.recoder.LE_InsStat,
            "clobber": self.default_parameters["lerecon"]["clobber"],
            "history": self.default_parameters["lerecon"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def legtigen(self, **kwargs):
        default_parameters = {
            "evtfile": "NONE",
            "instatusfile": self.recoder.LE_InsStat,
            "tempfile": self.recoder.LE_TH,
            "ehkfile": self.recoder.LE_EHK,
            "outfile": self.dner.legtigen["outfile"],
            "defaultexpr": self.default_parameters["legtigen"]["defaultexpr"],
            "expr": self.default_parameters["legtigen"]["expr"],
            "clobber": self.default_parameters["legtigen"]["clobber"],
            "history": self.default_parameters["legtigen"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def legticorr(self, **kwargs):
        if kwargs["mode"] == "command":
            self.logger.warning(
                "If 'mode' is set to 'command', the 'sigma' and 'dtime' keywords will not be effective."
            )
        default_parameters = {
            "reconfile": self.get(self.recoder.status["recon"]),
            "oldgti": self.get(self.recoder.status["gtitmp"]),
            "newgti": self.dner.legticorr["newgti"],
            "mode": "import",
            "sigma": 2.5,
            "dtime": 0.0,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lescreen(self, **kwargs):
        # 这里建议生成的screen文件是使用的GTI未经切割的
        default_parameters = {
            "evtfile": self.get(self.recoder.status["recon"]),
            "gtifile": self.get(self.recoder.status["gti"]),
            "outfile": self.dner.lescreen["outfile"],
            "userdetid": self.default_parameters["lescreen"]["userdetid"],
            "starttime": self.default_parameters["lescreen"]["starttime"],
            "stoptime": self.default_parameters["lescreen"]["stoptime"],
            "minPI": self.default_parameters["lescreen"]["minPI"],
            "maxPI": self.default_parameters["lescreen"]["maxPI"],
            "clobber": self.default_parameters["lescreen"]["clobber"],
            "history": self.default_parameters["lescreen"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lelcgen(self, **kwargs):
        # 额外返回(minE, maxE, timedel)
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "LE")
            maxPI = energy_to_pi(kwargs["maxE"], "LE")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"lelcgen: Apply the user energy range: {minE}-{maxE} keV"
            )
        else:
            minPI = self.default_parameters["lelcgen"]["minPI"]
            maxPI = self.default_parameters["lelcgen"]["maxPI"]
            minE = pi_to_energy(minPI, "LE")[0]
            maxE = pi_to_energy(maxPI, "LE")[-1]
            self.logger.debug(
                f"lelcgen: Apply the default minE and maxE: {minE}, {maxE}"
            )

        if "binsize" in kwargs.keys():
            timedel = matching_time_unit(kwargs["binsize"])
            self.logger.debug(f'lelcgen: Apply the user binsize: {kwargs["binsize"]}')
        else:
            timedel = matching_time_unit(self.default_parameters["lelcgen"]["binsize"])
            self.logger.debug(
                f'lelcgen: Apply the default binsize: {self.default_parameters["lelcgen"]["binsize"]}'
            )

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.lelcgen(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile_prefix"
            ],
            "userdetid": self.default_parameters["lelcgen"]["userdetid"],
            "binsize": self.default_parameters["lelcgen"]["binsize"],
            "starttime": self.default_parameters["lelcgen"]["starttime"],
            "stoptime": self.default_parameters["lelcgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "eventtype": self.default_parameters["lelcgen"]["eventtype"],
            "clobber": self.default_parameters["lelcgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def lebkgmap_lc(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcfile = self.get(node_name=lcnode)
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(lcfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.flush()

        # 能段信息和时间分辨都从lcraw中获取
        minPI = self.recoder.get_graph()[lcnode]["minPI"]
        maxPI = self.recoder.get_graph()[lcnode]["maxPI"]
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "ftype": self.default_parameters["lebkgmap_lc"]["ftype"],
            "evtfile": self.get(self.recoder.status["screen"]),
            "gtifile": self.get(self.recoder.status["gti"]),
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.lebkgmap_lc(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile_prefix"
            ],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lenetlcgen(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcbkgnode = self.recoder.status["lcbkg"]

        # 能段信息和时间分辨都从lcraw中获取
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "lcraw": self.get(lcnode),
            "lcbkg": self.get(lcbkgnode),
            "outfile": self.dner.lenetlcgen(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile"
            ],
        }

        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def lenetpdsgen(self, **kwargs):
        """
        所有参数都支持以float或str的格式输入，会自动转换为最适合的方式
        """
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "leahy",
            "outfile": self.dner.lenetpdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": False,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lermspdsgen(self, **kwargs):
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "rms",
            "outfile": self.dner.lermspdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": True,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lespecgen(self, **kwargs):
        """
        当指定starttime 和 stoptime 后，会生成一个临时gti文件用于背景生成时读
        取，目的是同步曝光时间；如果没有指定时间，则gti属性默认为标准gti
        """
        # 一般生成全能段
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "LE")
            maxPI = energy_to_pi(kwargs["maxE"], "LE")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"lespecgen: Apply the user energy range: {minE}-{maxE}keV"
            )
        else:
            minPI = self.default_parameters["lelcgen"]["minPI"]
            maxPI = self.default_parameters["lelcgen"]["maxPI"]
            minE = pi_to_energy(minPI, "LE")[0]
            maxE = pi_to_energy(maxPI, "LE")[-1]
            self.logger.debug(
                f"lespecgen: Apply the default energy range: {minE}-{maxE} keV"
            )

        if "starttime" and "stoptime" in kwargs.keys():
            # 如果指定了时间范围，生成对应范围的临时gti
            gtihdul = generate_GTI(
                [[kwargs["starttime"], kwargs["stoptime"]]],
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".fits", dir="/tmp"
            ) as tmpfile:
                gtihdul.writeto(tmpfile.name, overwrite=True)
                gti = tmpfile.name
        else:
            # 如果没有指定时间范围，读取标准gti
            gti = self.get(self.recoder.status["gti"])

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.lespecgen(minE=minE, maxE=maxE)[
                "outfile_prefix"
            ],  # 注意，这里的outfile只是前缀
            "userdetid": self.default_parameters["lespecgen"]["userdetid"],
            "starttime": self.default_parameters["lespecgen"]["starttime"],
            "stoptime": self.default_parameters["lespecgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "eventtype": self.default_parameters["lespecgen"]["eventtype"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "gti": gti}

    def lerspgen(self, **kwargs):

        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径

        # 能段信息从spec中获取
        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "phafile": specfile,
            "outfile": self.dner.lerspgen(minE, maxE)["outfile"],
            "attfile": self.recoder.LE_Att,
            "tempfile": self.recoder.LE_TH,
            "ra": self.default_parameters["lerspgen"]["ra"],
            "dec": self.default_parameters["lerspgen"]["dec"],
            "clobber": self.default_parameters["lerspgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def lebkgmap_spec(self, **kwargs):
        """
        lebkgmap同时读取了screen文件和GTI文件，但是不会读取evtfile中的好时间间隔
        表，所以最终的背景文件会以GTI文件中的好时间间隔表进行事件过滤和文件生成
        """
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(specfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.flush()

        # 能段信息从spec中获取
        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "ftype": self.default_parameters["lebkgmap_spec"]["ftype"],
            "evtfile": self.get(self.recoder.status["screen"]),
            "gtifile": self.recoder.get_attr(
                node=self.recoder.status["spec"], attr="gti"
            ),  # 这里GTI必须和spec的GTI一致
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.lebkgmap_spec(minE, maxE)[
                "outfile_prefix"
            ],  # 这里只是文件prefix
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def grppha_le(self, **kwargs):
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径

        # 能段信息从spec中获取
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "infile": specfile,
            "outfile": self.dner.grppha_le(minE, maxE)["outfile"],
            "group": "min 20",
            "respfile": self.get(self.recoder.status["rsp"]),
            "backfile": self.get(self.recoder.status["specbkg"]),
            "clobber": "yes",
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters


class MEParameters(BaseParameters):

    def __init__(self, recoder, exppath, logger):
        super().__init__(recoder, exppath, logger)
        # self.dirManager = PathGenerator(exppath, dti="LE")
        # self.fnameManager = FilenameGenerator(exppath, dti="LE")

        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = pathjoin(dir_path, "..", "json", "me_parameters.json")
        with open(json_path, "r") as file:
            self.default_parameters = json.load(file)

    def mepical(self, **kwargs):
        default_parameters = {
            "evtfile": self.recoder.ME_Evt,
            "tempfile": self.recoder.ME_TH,
            "outfile": self.dner.mepical["outfile"],
            "clobber": self.default_parameters["mepical"]["clobber"],
            "history": self.default_parameters["mepical"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def megrade(self, **kwargs):
        # 将产生两个文件，outfile和deadfile
        default_parameters = {
            "evtfile": self.get(self.recoder.status["pi"]),
            "outfile": self.dner.megrade["outfile"],
            "deadfile": self.dner.megrade["deadfile"],
            "binsize": self.default_parameters["megrade"]["binsize"],
            "clobber": self.default_parameters["megrade"]["clobber"],
            "history": self.default_parameters["megrade"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def megtigen(self, **kwargs):
        default_parameters = {
            "tempfile": self.recoder.ME_TH,
            "ehkfile": self.recoder.ME_EHK,
            "outfile": self.dner.megtigen["outfile"],
            "defaultexpr": self.default_parameters["megtigen"]["defaultexpr"],
            "expr": self.default_parameters["megtigen"]["expr"],
            "clobber": self.default_parameters["megtigen"]["clobber"],
            "history": self.default_parameters["megtigen"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def megticorr(self, **kwargs):
        # 将产生两个文件，newgti和baddetfile(文件名中有stauts)
        detectorstatus = pathjoin(
            os.getenv("HEADAS"), "refdata", "medetectorstatus.fits"
        )
        default_parameters = {
            "gradefile": self.get(self.recoder.status["grade"]),
            "oldgti": self.get(self.recoder.status["gtitmp"]),
            "newgti": self.dner.megticorr["newgti"],
            "detectorstatus": detectorstatus,
            "baddetfile": self.dner.megticorr["baddetfile"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def mescreen(self, **kwargs):
        default_parameters = {
            "evtfile": self.get(self.recoder.status["grade"]),
            "gtifile": self.get(self.recoder.status["gti"]),
            "outfile": self.dner.mescreen["outfile"],
            "baddetfile": self.get(self.recoder.status["baddet"]),
            "userdetid": self.default_parameters["mescreen"]["userdetid"],
            "starttime": self.default_parameters["mescreen"]["starttime"],
            "stoptime": self.default_parameters["mescreen"]["stoptime"],
            "minPI": self.default_parameters["mescreen"]["minPI"],
            "maxPI": self.default_parameters["mescreen"]["maxPI"],
            "clobber": self.default_parameters["mescreen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def melcgen(self, **kwargs):
        # 额外返回(minE, maxE, timedel)
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "ME")
            maxPI = energy_to_pi(kwargs["maxE"], "ME")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"melcgen: Apply the user energy range: {minE}-{maxE} keV"
            )
        else:
            minPI = self.default_parameters["melcgen"]["minPI"]
            maxPI = self.default_parameters["melcgen"]["maxPI"]
            minE = pi_to_energy(minPI, "ME")[0]
            maxE = pi_to_energy(maxPI, "ME")[-1]
            self.logger.debug(
                f"melcgen: Apply the default energy range: {minE}-{maxE}keV"
            )

        if "binsize" in kwargs.keys():
            timedel = matching_time_unit(kwargs["binsize"])
            self.logger.debug(f'melcgen: Apply the user binsize: {kwargs["binsize"]}')
        else:
            timedel = matching_time_unit(self.default_parameters["melcgen"]["binsize"])
            self.logger.debug(
                f'melcgen: Apply the default binsize: {self.default_parameters["melcgen"]["binsize"]}'
            )

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.melcgen(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile_prefix"
            ],
            "deadfile": self.get(self.recoder.status["deadf"]),
            "userdetid": self.default_parameters["melcgen"]["userdetid"],
            "binsize": self.default_parameters["melcgen"]["binsize"],
            "starttime": self.default_parameters["melcgen"]["starttime"],
            "stoptime": self.default_parameters["melcgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "deadcorr": self.default_parameters["melcgen"]["deadcorr"],
            "clobber": self.default_parameters["melcgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def mebkgmap_lc(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcfile = self.get(node_name=lcnode)
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(lcfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.close

        # 能段信息和时间分辨都从lcraw中获取
        minPI = self.recoder.get_graph()[lcnode]["minPI"]
        maxPI = self.recoder.get_graph()[lcnode]["maxPI"]
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "ftype": "lc",
            "evtfile": self.get(self.recoder.status["screen"]),
            "ehkfile": self.recoder.ME_EHK,
            "gtifile": self.get(self.recoder.status["gti"]),
            "dtimefile": self.get(self.recoder.status["deadf"]),
            "thfile": self.recoder.ME_TH,
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.mebkgmap_lc(minE, maxE, timedel)["outfile_prefix"],
            "statusfile": self.get(self.recoder.status["baddet"]),
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def menetlcgen(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcbkgnode = self.recoder.status["lcbkg"]

        # 能段信息和时间分辨都从lcraw中获取
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "lcraw": self.get(lcnode),
            "lcbkg": self.get(lcbkgnode),
            "outfile": self.dner.menetlcgen(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile"
            ],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def menetpdsgen(self, **kwargs):
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "leahy",
            "outfile": self.dner.menetpdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": False,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def mermspdsgen(self, **kwargs):
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "rms",
            "outfile": self.dner.mermspdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": True,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def mespecgen(self, **kwargs):
        """
        文档同lespecgen
        """
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "ME")
            maxPI = energy_to_pi(kwargs["maxE"], "ME")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"mespecgen: Apply the user energy range: {minE}-{maxE} keV"
            )
        else:
            minPI = self.default_parameters["mespecgen"]["minPI"]
            maxPI = self.default_parameters["mespecgen"]["maxPI"]
            minE = pi_to_energy(minPI, "ME")[0]
            maxE = pi_to_energy(maxPI, "ME")[-1]
            self.logger.debug(
                f"mespecgen: Apply the default energy range: {minE}-{maxE}keV"
            )

        if "starttime" and "stoptime" in kwargs.keys():
            gtihdul = generate_GTI(
                [[kwargs["starttime"], kwargs["stoptime"]]],
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".fits", dir="/tmp"
            ) as tmpfile:
                gtihdul.writeto(tmpfile.name, overwrite=True)
                gti = tmpfile.name
        else:
            # 如果没有指定时间范围，读取标准gti
            gti = self.get(self.recoder.status["gti"])

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.mespecgen(minE, maxE)[
                "outfile_prefix"
            ],  # 这里outfile只是前缀
            "deadfile": self.get(self.recoder.status["deadf"]),
            "userdetid": self.default_parameters["mespecgen"]["userdetid"],
            "starttime": self.default_parameters["mespecgen"]["starttime"],
            "stoptime": self.default_parameters["mespecgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "clobber": self.default_parameters["mespecgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "gti": gti}

    def merspgen(self, **kwargs):

        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)

        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "phafile": specfile,
            "outfile": self.dner.merspgen(minE, maxE)["outfile"],
            "attfile": self.recoder.ME_Att,
            "ra": self.default_parameters["merspgen"]["ra"],
            "dec": self.default_parameters["merspgen"]["dec"],
            "clobber": self.default_parameters["merspgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def mebkgmap_spec(self, **kwargs):
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(specfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.flush()

        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "ftype": "spec",
            "evtfile": self.get(self.recoder.status["screen"]),
            "ehkfile": self.recoder.ME_EHK,
            "gtifile": self.recoder.get_attr(
                node=self.recoder.status["spec"], attr="gti"
            ),  # 这里GTI必须和spec的GTI一致
            "dtimefile": self.get(self.recoder.status["deadf"]),
            "thfile": self.recoder.ME_TH,
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.mebkgmap_spec(minE, maxE)["outfile_prefix"],
            "statusfile": self.get(self.recoder.status["baddet"]),
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def grppha_me(self, **kwargs):
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径

        # 能段信息从spec中获取
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "infile": specfile,
            "outfile": self.dner.grppha_me(minE, maxE)["outfile"],
            "group": "min 20",
            "respfile": self.get(self.recoder.status["rsp"]),
            "backfile": self.get(self.recoder.status["specbkg"]),
            "clobber": "yes",
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters


class HEParameters(BaseParameters):

    def __init__(self, recoder, exppath, logger):
        super().__init__(recoder, exppath, logger)
        # self.dirManager = PathGenerator(exppath, dti="LE")
        # self.fnameManager = FilenameGenerator(exppath, dti="LE")

        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = pathjoin(dir_path, "..", "json", "he_parameters.json")
        with open(json_path, "r") as file:
            self.default_parameters = json.load(file)

    def hepical(self, **kwargs):
        default_parameters = {
            "evtfile": self.recoder.HE_Evt,
            "outfile": self.dner.hepical["outfile"],
            "minpulsewidth": self.default_parameters["hepical"]["minpulsewidth"],
            "maxpulsewidth": self.default_parameters["hepical"]["maxpulsewidth"],
            "clobber": self.default_parameters["hepical"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def hegtigen(self, **kwargs):
        default_parameters = {
            "hvfile": self.recoder.HE_HV,
            "tempfile": self.recoder.HE_TH,
            "pmfile": self.recoder.HE_PM,
            "outfile": self.dner.hegtigen["outfile"],
            "ehkfile": self.recoder.HE_EHK,
            "defaultexpr": self.default_parameters["hegtigen"]["defaultexpr"],
            "expr": self.default_parameters["hegtigen"]["expr"],
            "pmexpr": self.default_parameters["hegtigen"]["pmexpr"],
            "clobber": self.default_parameters["hegtigen"]["clobber"],
            "history": self.default_parameters["hegtigen"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def hescreen(self, **kwargs):
        default_parameters = {
            "evtfile": self.get(self.recoder.status["pi"]),
            "gtifile": self.get(self.recoder.status["gti"]),
            "outfile": self.dner.hescreen["outfile"],
            "userdetid": self.default_parameters["hescreen"]["userdetid"],
            "eventtype": self.default_parameters["hescreen"]["eventtype"],
            "anticoincidence": self.default_parameters["hescreen"]["anticoincidence"],
            "starttime": self.default_parameters["hescreen"]["starttime"],
            "stoptime": self.default_parameters["hescreen"]["stoptime"],
            "minPI": self.default_parameters["hescreen"]["minPI"],
            "maxPI": self.default_parameters["hescreen"]["maxPI"],
            "clobber": self.default_parameters["hescreen"]["clobber"],
            "history": self.default_parameters["hescreen"]["history"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def helcgen(self, **kwargs):
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "HE")
            maxPI = energy_to_pi(kwargs["maxE"], "HE")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"helcgen: Apply the user energy range: {minE}-{maxE} keV"
            )
        else:
            minPI = self.default_parameters["helcgen"]["minPI"]
            maxPI = self.default_parameters["helcgen"]["maxPI"]
            minE = pi_to_energy(minPI, "HE")[0]
            maxE = pi_to_energy(maxPI, "HE")[-1]
            self.logger.debug(
                f"helcgen: Apply the default energy range: {minE}-{maxE}keV"
            )

        if "binsize" in kwargs.keys():
            timedel = matching_time_unit(kwargs["binsize"])
            self.logger.debug(f'helcgen: Apply the user binsize: {kwargs["binsize"]}')
        else:
            timedel = matching_time_unit(self.default_parameters["helcgen"]["binsize"])
            self.logger.debug(
                f'helcgen: Apply the default binsize: {self.default_parameters["helcgen"]["binsize"]}'
            )

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.helcgen(minE, maxE, timedel)["outfile_prefix"],
            "deadfile": self.recoder.HE_DTime,
            "userdetid": self.default_parameters["helcgen"]["userdetid"],
            "binsize": self.default_parameters["helcgen"]["binsize"],
            "starttime": self.default_parameters["helcgen"]["starttime"],
            "stoptime": self.default_parameters["helcgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "deadcorr": self.default_parameters["helcgen"]["deadcorr"],
            "clobber": self.default_parameters["helcgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def hebkgmap_lc(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcfile = self.get(node_name=lcnode)
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(lcfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.flush()

        # 能段信息和时间分辨都从lcraw中获取
        minPI = self.recoder.get_graph()[lcnode]["minPI"]
        maxPI = self.recoder.get_graph()[lcnode]["maxPI"]
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "ftype": "lc",
            "evtfile": self.get(self.recoder.status["screen"]),
            "ehkfile": self.recoder.HE_EHK,
            "gtifile": self.get(self.recoder.status["gti"]),
            "dtimefile": self.recoder.HE_DTime,
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.hebkgmap_lc(minE, maxE, timedel)["outfile_prefix"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def henetlcgen(self, **kwargs):
        lcnode = self.recoder.status["lcraw"]
        lcbkgnode = self.recoder.status["lcbkg"]

        # 能段信息和时间分辨都从lcraw中获取
        minE = self.recoder.get_graph()[lcnode]["minE"]
        maxE = self.recoder.get_graph()[lcnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnode]["timedel"]

        default_parameters = {
            "lcraw": self.get(lcnode),
            "lcbkg": self.get(lcbkgnode),
            "outfile": self.dner.henetlcgen(minE=minE, maxE=maxE, timedel=timedel)[
                "outfile"
            ],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "timedel": timedel}

    def henetpdsgen(self, **kwargs):
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "leahy",
            "outfile": self.dner.henetpdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": False,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def hermspdsgen(self, **kwargs):
        lcnetnode = self.recoder.status["lcnet"]

        minE = self.recoder.get_graph()[lcnetnode]["minE"]
        maxE = self.recoder.get_graph()[lcnetnode]["maxE"]
        timedel = self.recoder.get_graph()[lcnetnode]["timedel"]
        # 目前不支持在功率谱阶段调整timedel

        rebin = kwargs["rebin"]
        segment = kwargs["segment"]

        default_parameters = {
            "lcnet": self.get(lcnetnode),
            "segment": segment,
            "rebin": rebin,
            "norm": "rms",
            "outfile": self.dner.hermspdsgen(
                segment=segment, minE=minE, maxE=maxE, timedel=timedel, rebin=rebin
            )["outfile"],
            "subtracted_white_noise": True,
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def hespecgen(self, **kwargs):
        """
        参考lespecgen的文档
        """
        if "minE" and "maxE" in kwargs.keys():
            minPI = energy_to_pi(kwargs["minE"], "HE")
            maxPI = energy_to_pi(kwargs["maxE"], "HE")
            minE = kwargs["minE"]
            maxE = kwargs["maxE"]
            self.logger.debug(
                f"hespecgen: Apply the user energy range: {minE}-{maxE}keV"
            )
        else:
            minPI = self.default_parameters["helcgen"]["minPI"]
            maxPI = self.default_parameters["helcgen"]["maxPI"]
            minE = pi_to_energy(minPI, "HE")[0]
            maxE = pi_to_energy(maxPI, "HE")[-1]
            self.logger.debug(
                f"hespecgen: Apply the default energy range: {minE}-{maxE} keV"
            )

        if "starttime" and "stoptime" in kwargs.keys():
            # 如果指定了时间范围，生成对应范围的临时gti(生成背景使用)
            gtihdul = generate_GTI(
                [[kwargs["starttime"], kwargs["stoptime"]]],
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".fits", dir="/tmp"
            ) as tmpfile:
                gtihdul.writeto(tmpfile.name, overwrite=True)
                gti = tmpfile.name
        else:
            # 如果没有指定时间范围，读取标准gti
            gti = self.get(self.recoder.status["gti"])

        default_parameters = {
            "evtfile": self.get(self.recoder.status["screen"]),
            "outfile": self.dner.hespecgen(minE, maxE)["outfile_prefix"],
            "deadfile": self.recoder.HE_DTime,
            "userdetid": self.default_parameters["hespecgen"]["userdetid"],
            "eventtype": self.default_parameters["hespecgen"]["eventtype"],
            "starttime": self.default_parameters["hespecgen"]["starttime"],
            "stoptime": self.default_parameters["hespecgen"]["stoptime"],
            "minPI": minPI,
            "maxPI": maxPI,
            "clobber": self.default_parameters["hespecgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters, {"minE": minE, "maxE": maxE, "gti": gti}

    def herspgen(self, **kwargs):

        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径

        # 能段信息从spec中获取
        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "phafile": specfile,
            "outfile": self.dner.herspgen(minE, maxE)["outfile"],
            "attfile": self.recoder.HE_Att,
            "ra": self.default_parameters["herspgen"]["ra"],
            "dec": self.default_parameters["herspgen"]["dec"],
            "clobber": self.default_parameters["herspgen"]["clobber"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def hebkgmap_spec(self, **kwargs):
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp:
            temp.write(specfile)
            temp.write("\n")  # 最后必须存在一个换行符
            temp.flush()

        # 能段信息从spec中获取
        minPI = self.recoder.get_graph()[specnode]["minPI"]
        maxPI = self.recoder.get_graph()[specnode]["maxPI"]
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "ftype": "spec",
            "evtfile": self.get(self.recoder.status["screen"]),
            "ehkfile": self.recoder.HE_EHK,
            "gtifile": self.recoder.get_attr(
                node=self.recoder.status["spec"], attr="gti"
            ),
            "dtimefile": self.recoder.HE_DTime,
            "fascii": temp.name,
            "minPI": minPI,
            "maxPI": maxPI,
            "outfile": self.dner.hebkgmap_spec(minE, maxE)["outfile_prefix"],
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters

    def grppha_he(self, **kwargs):
        specnode = self.recoder.status["spec"]
        specfile = self.get(node_name=specnode)  # 谱文件完整路径

        # 能段信息从spec中获取
        minE = self.recoder.get_graph()[specnode]["minE"]
        maxE = self.recoder.get_graph()[specnode]["maxE"]

        default_parameters = {
            "infile": specfile,
            "outfile": self.dner.grppha_he(minE, maxE)["outfile"],
            "group": "min 20",
            "respfile": self.get(self.recoder.status["rsp"]),
            "backfile": self.get(self.recoder.status["specbkg"]),
            "clobber": "yes",
        }
        default_parameters = update_parameters(default_parameters, **kwargs)
        return default_parameters
