import os
import re
import time
from typing import Literal

import stingray
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table, QTable
from uncertainties import ufloat, unumpy
from stingray import Powerspectrum, AveragedPowerspectrum

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..core.execute import CommandExecutor, gen_cmd_string
from .lcutils import open_lc

mpl.rcParams["axes.linewidth"] = 2.0
mpl.rcParams["legend.fontsize"] = 20.0
mpl.rcParams["legend.title_fontsize"] = 20.0
mpl.rcParams["axes.labelsize"] = 25.0
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.major.size"] = 8.0
mpl.rcParams["xtick.minor.size"] = 4.0
mpl.rcParams["xtick.major.width"] = 1.6
mpl.rcParams["xtick.minor.width"] = 1.6
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.major.size"] = 8.0
mpl.rcParams["ytick.minor.size"] = 4.0
mpl.rcParams["ytick.major.width"] = 1.6
mpl.rcParams["ytick.minor.width"] = 1.6
mpl.rcParams["xtick.labelsize"] = 20.0
mpl.rcParams["ytick.labelsize"] = 20.0
mpl.rcParams["savefig.bbox"] = "tight"

mpl.rcParams["axes.prop_cycle"] = cycler(
    "color",
    [
        "#FA9D3A",
        "#D7566B",
        "#515151",
        "#9B179E",
        "#7200A8",
        "#45029E",
        "#0C0786",
        "#000000",
    ],
)


def open_pds(fps, dtype: Literal["stingray", "tuple"] = "stingray"):
    """
    从FITS文件中读取功率谱数据
    """

    def stglize(freq, freq_err, power, power_err):
        stg = Powerspectrum()
        stg.freq = freq
        stg.freq_err = freq_err
        stg.power = power
        stg.power_err = power_err
        return stg

    if isinstance(fps, str):
        pds = Table.read(fps, 1, unit_parse_strict="silent")
        freq = pds["FREQUENCY"]
        freq_err = pds["XAX_E"]
        power = pds["POWER"]
        power_err = pds["ERROR"]
        if dtype == "tuple":
            return (freq, freq_err, power, power_err)
        elif dtype == "stingray":
            return stglize(freq, freq_err, power, power_err)

    elif isinstance(fps, Powerspectrum):
        if dtype == "tuple":
            try:
                res = fps.freq, None, fps.power, fps.power_err
            except AttributeError:
                res = fps.freq, None, fps.power, None
            return res
        elif dtype == "stingray":
            return fps


def generate_pds_from_lc(
    lc,
    segment=256.0,
    rebin=0,
    norm: Literal["leahy", "rms"] = "leahy",
    subtracted_white_noise=False,
    blindfps=None,
    logger=None,
):
    """
    从光变曲线生成平均功率谱，返回Powerspectrum对象
    """
    lcobj = open_lc(lc)
    meanrate = lcobj.meanrate

    # 判断segment是否超过GTI最大间隔
    gti = lcobj.gti
    max_gti_period = np.max(gti[:, -1] - gti[:, 0])
    while segment > max_gti_period:
        if bool(logger):
            logger.warning(
                f"Segment is too large, which will be reduced to {int(segment / 2)}s!"
            )
        segment = segment / 2

    # 功率谱
    try:
        # 这里只生成leahy归一的功率谱
        pdsobj = AveragedPowerspectrum(
            lcobj, gti=lcobj.gti, segment_size=segment, norm="leahy"
        )
        pdsobj.meanrate = meanrate
    except RuntimeError as e:
        if logger:
            logger.debug("RuntimeError in AveragedPowerspectrum.")
    except:
        pass

    # 重分箱
    if rebin:
        pdsobj = rebin_pds(pdsobj, rebin)

    # 去除白噪声和RMS归一
    if subtracted_white_noise:
        if norm == "leahy":
            pdsobj = subtracted_white_noise_pds(pdsobj, norm="leahy", blinefps=blindfps)
        elif norm == "rms":
            pdsobj = subtracted_white_noise_pds(
                pdsobj, meanrate=meanrate, norm="rms", blindfps=blindfps
            )
    else:
        if norm == "leahy":
            pass
        elif norm == "rms":
            raise Exception("RMS normalization must be subtracted white noise!")

    return pdsobj


def subtracted_white_noise_pds(fps, meanrate=None, norm="leahy", blindfps=None):
    """
    仅支持输入leahy归一的功率谱
    """
    pdsobj = open_pds(fps)

    if blindfps:
        blinddf = open_pds(blindfps).to_pandas()
        leahy_white_noise_level = np.mean(blinddf[blinddf["freq"] >= 100.0]["power"])
    else:
        pdsdf = pdsobj.to_pandas()
        leahy_white_noise_level = np.mean(pdsdf[pdsdf["freq"] >= 100.0]["power"])

    if norm == "leahy":
        pdsobj.power -= leahy_white_noise_level
        pdsobj.leahy_white_noise_level = leahy_white_noise_level
    elif norm == "rms":
        pdsobj.power -= leahy_white_noise_level
        pdsobj.power /= meanrate
        pdsobj.power_err /= meanrate
        pdsobj.leahy_white_noise_level = leahy_white_noise_level / meanrate
        pdsobj.norm = "frac"

    return pdsobj


def rebin_pds(fps, rebin):
    """
    重分箱功率谱，返回Powerspectrum对象
    当rebin大于0时， 为线性分箱； 当rebin小于0时， 为对数分箱；
    当rebin的绝对值大于1时，为POWERSPEC格式
    """
    pdsobj = open_pds(fps)

    if rebin > 0:
        if rebin > 1:
            pdsobj = pdsobj.rebin(rebin - 1)  # POWERSPEC 格式
        else:
            pdsobj = pdsobj.rebin(rebin)  # Stingray 格式
    elif rebin < 0:
        if rebin < -1:
            pdsobj = pdsobj.rebin_log(abs(rebin + 1))  # POWERSPEC 格式
        else:
            pdsobj = pdsobj.rebin_log(abs(rebin))  # Stingray 格式

    return pdsobj


def plotpds(fps, rebin=0):
    """
    绘制功率谱， 返回(fig,ax)
    """
    obj = open_pds(fps, dtype="stingray")

    if abs(rebin) > 2:
        raise Exception("rebin must be within -1 to 1")
    if rebin > 0:
        if rebin > 1:
            obj = obj.rebin(rebin - 1)  # POWERSPEC format
        else:
            obj = obj.rebin(rebin)
    elif rebin < 0:
        if rebin < -1:
            obj = obj.rebin_log(abs(rebin + 1))  # POWERSPEC format
        else:
            obj = obj.rebin_log(abs(rebin))

    try:
        freq_err = obj.freq_err
    except AttributeError:
        freq_err = np.append(
            (np.array(obj.freq[1:]) - np.array(obj.freq[0:-1])) / 2,
            (obj.freq[-1] - obj.freq[-2]) / 2,
        )

    try:
        power_err = obj.power_err
    except AttributeError:
        power_err = None

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    if power_err is None:
        ax.step(x=obj.freq, y=obj.power, where="mid")
    else:
        ax.step(x=obj.freq, y=obj.power, where="mid", color="#FA9D3A")
        ax.errorbar(
            x=obj.freq, y=obj.power, yerr=obj.power_err, linestyle="", color="#FA9D3A"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    if isinstance(fps, str):
        ax.set_title(fps)

    if freq_err is None:
        x_lower = obj.freq[0]
        x_upper = obj.freq[-1]
    else:
        x_lower = obj.freq[0] - freq_err[0]
        x_upper = obj.freq[-1] + freq_err[-1]
    ax.set_xlim(x_lower, x_upper)

    if obj.power_err is None:
        y_lower = np.min(obj.power)
        y_upper = np.max(obj.power)
    else:
        y_lower = np.min(obj.power) - obj.power_err[np.argmin(obj.power)]
        y_upper = np.max(obj.power) + obj.power_err[np.argmax(obj.power)]
        ax.set_ylim(y_lower, y_upper)

    return fig, ax


def meanpds(*pdsfiles, method: Literal["hdul", "tb"] = "hdul"):
    """
    输入多个功率谱文件，生成他们的平均功率谱
    """
    all_powers = []
    pdsobj = fits.open(pdsfiles[0])
    for fits_file in pdsfiles:
        frequency, frequency_err, power, power_err = open_pds(fits_file, dtype="tuple")
        # 将功率及其误差转换为uncertainties库可以处理的形式
        power_with_err = unumpy.uarray(power, power_err)
        all_powers.append(power_with_err)

    # 计算平均功率谱
    mean_power = np.mean(unumpy.nominal_values(all_powers), axis=0)
    mean_error = np.sqrt(np.sum(unumpy.std_devs(all_powers) ** 2, axis=0)) / len(
        all_powers
    )
    if method == "tb":
        return (frequency, frequency_err, mean_power, mean_error)
    elif method == "hdul":
        pdstb = QTable(
            data=[
                frequency.tolist(),
                frequency_err.tolist(),
                mean_power.tolist(),
                mean_error.tolist(),
            ],
            names=["FREQUENCY", "XAX_E", "POWER", "ERROR"],
            dtype=[np.float64, np.float64, np.float64, np.float64],
        )
        pdsobj[1] = fits.BinTableHDU(data=pdstb, header=pdsobj[1].header)
        return pdsobj


def meanpds_name_gen(*fpses):
    """
    生成平均功率谱的文件名
    """
    pattern = re.compile(r".*/P(\d+)_([A-Z]+)_512s_rms_(\d+-\d+)keV_2msBin_PDS.fps")

    observation_id = ""
    segment = "512s"
    rms = "rms"
    time_bin = "2msBin"
    energy_ranges = []

    for filename in fpses:
        match = pattern.match(filename)
        if match:
            if not observation_id:
                observation_id = match.group(1)
            energy_start, energy_end = map(int, match.group(3).split("-"))
            energy_ranges.append((energy_start, energy_end))

    # 合并能段范围
    min_energy = min(energy_ranges, key=lambda x: x[0])[0]
    max_energy = max(energy_ranges, key=lambda x: x[1])[1]
    energy_range_str = f"{min_energy}-{max_energy}keV"

    # 平均功率谱的文件名
    mean_pds_filename = (
        f"P{observation_id}_MEAN_{segment}_{rms}_{energy_range_str}_{time_bin}_PDS.fps"
    )

    return mean_pds_filename


def calculate_rms(x, *y, fmt="euf"):
    """
    计算模型的RMS；如果传入多个模型，将计算最大轮廓的RMS
    """
    ay = np.maximum(y)
    if fmt == "euf":
        return np.trapz(ay / x, x) ** 0.5
    elif fmt == "uf":
        return np.trapz(ay, x) ** 0.5
    else:
        raise ValueError("Unknown fmt: {}".format(fmt))


def save_as_file(
    fps,
    filename="output.fps",
):
    """
    将 PowerSpectrum 保存为标准fits文件
    """
    pds = open_pds(fps)
    # 计算频率误差 XAX_E
    freq = pds.freq
    xax_e = (freq[1:] - freq[:-1]) / 2
    xax_e = np.append(xax_e, (freq[-1] - freq[-2]) / 2)  # 最后一个值复制前一个

    # 创建 BinTableHDU 中的数据
    col1 = fits.Column(name="FREQUENCY", format="D", array=freq)
    col2 = fits.Column(name="XAX_E", format="D", array=xax_e)
    col3 = fits.Column(name="POWER", format="D", array=pds.power)
    col4 = fits.Column(name="ERROR", format="D", array=pds.power_err)

    cols = fits.ColDefs([col1, col2, col3, col4])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.name = "RESULTS"

    # 填充 Header 数据
    header = hdu.header
    header["EXTNAME"] = "RESULTS"
    header["CREATOR"] = "Stingray {}".format(stingray.__version__)
    header["HDUCLASS"] = "OGIP"
    header["HDUCLAS1"] = "TEMPORALDATA"
    header["HDUCLAS2"] = "POWER SPECTRA"
    header["HDUCLAS3"] = "RESULTS"
    header["CONTENT"] = "POWER SPECTRA"
    header["ORIGIN"] = "HXMT PIPELINE"
    header["AUTHOR"] = "cxgao@shao.ac.cn"

    header_items = {
        "SEGMENT": pds.segment_size,
        "DT": pds.dt,
        "MAXFREQ": 1 / pds.dt / 2,
        "MINFREQ": 1 / pds.segment_size,
        "NORM": pds.norm,
        "MEANRATE": pds.meanrate,
        "AVGNUM": int(pds.m) if isinstance(pds.m, (int, float)) else pds.m[0],
    }
    for key, value in header_items.items():
        try:
            header[key] = value
        except:
            pass

    primary_hdu = fits.PrimaryHDU()

    # 创建 HDU list 并保存为 FITS 文件
    hdul = fits.HDUList([primary_hdu, hdu])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    hdul.writeto(filename, overwrite=True)

    print(f"Power spectrum saved as {filename}")


def fps2xsp(fpsfile):
    """
    将fps转化为xspec可读的形式
    """
    pwd = os.getcwd()
    if os.path.isabs(fpsfile):
        dirname = os.path.dirname(fpsfile)
        os.chdir(dirname)
    else:
        dirname = os.getcwd()
    basename = os.path.basename(fpsfile)
    prefix = basename.replace(".fps", "")

    freq, dfreq, power, dpower = open_pds(fpsfile, dtype="tuple")
    flux_data = np.array(
        [
            freq - dfreq,
            freq + dfreq,
            2 * power * dfreq,
            2 * dpower * dfreq,
        ]
    ).T
    flux_filename = f"{prefix}_flux.txt"
    np.savetxt(flux_filename, flux_data)

    pha_file = f"{prefix}.pha"
    rsp_file = f"{prefix}.rsp"

    params = {
        "infile": flux_filename,
        "phafile": pha_file,
        "rspfile": rsp_file,
        "clobber": "yes",
    }
    cmd_string = gen_cmd_string("flx2xsp", params, ktype="keyword")
    print(cmd_string)
    runner = CommandExecutor()
    runner.run(cmd_string)

    os.chdir(pwd)
    return os.path.join(dirname, pha_file)


def get_allpdsdata_fromxcm(xcmfile=None, savecsv=False, plottype="euf", savefmt=None):
    """
    输入xcm文件完整路径，返回绘图数据DataFrame
    """

    def xcm_restore(xcmfile):
        now_path = os.getcwd()
        os.chdir(os.path.dirname(xcmfile))
        xspec.Xset.restore(xcmfile)
        os.chdir(now_path)

    def str_restore(xspec_string):
        with open("tmp_lout.xcm", "w+") as f:
            f.write(xspec_string)
        xspec.Xset.restore("tmp_lout.xcm")
        os.remove("tmp_lout.xcm")
        return

    def auto_restore(lens, d):
        p = list(range(1, lens + 1))
        i1 = 0
        i2 = p.index(d)
        if i1 == i2:
            pass
        else:
            str_restore(f"delcomp {i1+1}-{i2}")
            p[:i2] = []
        i3 = 0
        i4 = len(p)
        if i4 == 1:
            pass
        else:
            str_restore(f"delcomp {i3+2}-{i4}")
            p[(i3 + 1) :] = []
        return

    # ------ main -------
    import xspec

    xcmfile_path = xcmfile
    os.chdir(os.path.dirname(xcmfile_path))
    xcm_restore(xcmfile_path)
    if savecsv:
        if not os.path.exists("PDS_MODEL"):
            os.mkdir("PDS_MODEL")
        os.chdir("PDS_MODEL")
        xspec.Plot.device = "ori_m.ps /vcps"
    else:
        xspec.Plot.device = "/null"
    xspec.Plot.xAxis = "keV"
    xspec.Plot(plottype)
    x = xspec.Plot.x()
    data = xspec.Plot.y()
    xerr = xspec.Plot.xErr()
    data_err = xspec.Plot.yErr()
    tol_m = xspec.Plot.model()

    if savecsv:
        xspec.Plot.device = "chi.ps /vcps"
    else:
        xspec.Plot.device = "/null"
    xspec.Plot("chi")
    chisq = xspec.Plot.y()

    m = xspec.AllModels(1)
    lens = len(m.componentNames)
    lor_M = []
    for i in list(range(1, lens + 1)):
        auto_restore(lens, i)
        if savecsv:
            xspec.Plot.device = f"lor{i}_m.ps /vcps"
        else:
            xspec.Plot.device = "/null"
        xspec.Plot(plottype)
        lor_M.append(xspec.Plot.model())
        xcm_restore(xcmfile_path)

    result_array = np.array([x, xerr, data, data_err, tol_m, chisq] + lor_M).T
    result = pd.DataFrame(
        result_array,
        columns=["x", "xerr", "data", "data_err", "model", "chisq"]
        + [f"lor{i}_m" for i in range(1, lens + 1)],
    )

    # 将结果保存成csv文件
    if savefmt:
        filename_lst = re.findall(savefmt, xcmfile_path)[-1]
    else:
        try:
            filename_lst = re.findall(
                r"P\d{12,}_(\w{2}).*(_[\.\-0-9]*keV).*", xcmfile_path
            )[-1]
        except:
            filename_lst = os.path.basename(xcmfile_path).split("_")[0]
    print(filename_lst)
    if isinstance(filename_lst, (list, tuple)):
        filename_lst = [i for i in filename_lst if i.strip() != ""]
        filename_prefix = "_".join(filename_lst)
        filename = "".join(filename_prefix) + ".csv"
    else:
        print(type(filename_lst))
        filename = filename_lst + ".csv"
    print(filename)
    if savecsv:
        result.to_csv(filename)

    return result


def plotpds_detile(
    csvfile,
    onlydata=False,
):
    """
    已废弃，用hxmt/pipeline/pdsutil.py中的plotpds替代
    """
    d = pd.read_csv(csvfile)
    title = os.path.basename(csvfile).split(".")[0]
    savepath = os.path.dirname(csvfile) + f"/{title}.pdf"
    x = d["x"]
    xerr = d["xerr"]
    data = d["data"]
    derr = d["data_err"]
    model = d["model"]
    chisq = d["chisq"]
    lors = d.iloc[:, 7:]

    # plt.rcParams['axes.linewidth'] = 2.0
    # plt.rcParams['xtick.major.size'] = 8.0
    # plt.rcParams['xtick.minor.size'] = 4.0
    # plt.rcParams['xtick.major.width'] = 1.2
    # plt.rcParams['xtick.minor.width'] = 1.2
    # plt.rcParams['ytick.major.size'] = 8.0
    # plt.rcParams['ytick.minor.size'] = 4.0
    # plt.rcParams['ytick.major.width'] = 1.2
    # plt.rcParams['ytick.minor.width'] = 1.2
    # plt.rcParams['xtick.labelsize'] = 20.0
    # plt.rcParams['ytick.labelsize'] = 20.0

    if onlydata:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_axes([0.0, 0.0] + [1.0, 1.0])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.errorbar(x=x, y=data, yerr=derr, xerr=xerr, fmt=".", color="black")
        ax1.set_xlim(x.min(), x.max())
        ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax1.set_ylabel(r"$Power$", fontsize=25)
        ax1.set_xlabel("Frequency (Hz)", fontsize=25)
        ax1.set_title(
            title,
            fontsize=25,
        )
        ymin, ymax = ax1.get_ybound()
        ax1.fill_between([0.0432, 0.0458], ymin, ymax, color="grey", alpha=0.6)
        ax1.vlines(
            x=0.0445, ymin=ymin, ymax=ymax, linestyles="--", color="red", linewidth=2.5
        )
        ax1.set_ylim(ymin, ymax)
        fig.savefig(savepath, bbox_inches="tight")
        return
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_axes([0.0, 0.4] + [1.0, 0.6])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.errorbar(x=x, y=data, yerr=derr, xerr=xerr, fmt=".", color="black")
    ax1.plot(x, model, "c-", linewidth=2.0, color="#1F77B4")
    yl, yr = ax1.get_ylim()
    for l in lors.values.T:
        ax1.plot(x, l, "--", color="grey")
    ax1.set_ylim(yl, yr)
    ax1.set_xlim(x.min(), x.max())
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelbottom=False,
    )
    ax1.set_ylabel(r"$Power\times Frequency (rms^2\times Hz)$", fontsize=17)
    # ax1.set_ylabel('rms', fontsize=25)
    ax1.set_title(
        title,
        fontsize=25,
    )

    ax2 = fig.add_axes([0, 0] + [1, 0.4], sharex=ax1)
    ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax2.step(x, chisq, where="mid", linewidth=2.0, color="#1F77B4")
    ax2.axhline(0, linestyle="--", color="grey")
    ax2.set_xlabel("Frequency (Hz)", fontsize=25)
    ax2.set_ylabel(r"$\Delta \chi^2$", fontsize=25)

    fig.savefig(savepath, bbox_inches="tight")

    return
