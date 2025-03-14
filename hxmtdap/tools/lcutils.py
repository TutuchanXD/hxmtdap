import os
from typing import Literal

import numpy as np
import pandas as pd
from stingray import Lightcurve
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table
import portion as P

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

from .brokenaxes import brokenaxes

mpl.rcParams["axes.linewidth"] = 2.0
mpl.rcParams["legend.fontsize"] = 16.0
mpl.rcParams["legend.title_fontsize"] = 20.0
mpl.rcParams["axes.labelsize"] = 16.0
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
mpl.rcParams["xtick.labelsize"] = 10.0
mpl.rcParams["ytick.labelsize"] = 10.0
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


def open_lc(lc, method="lc"):
    """
    用于打开lc文件，返回Lightcurve对象

    """
    if isinstance(lc, str):  # 文件名情况，总是使用astropy table
        # 判断hdu数量
        with fits.open(lc) as hdul:
            lens = len(hdul)

        tb = Table.read(lc, 1, unit_parse_strict="silent")
        if lens >= 3:
            gti = Table.read(lc, 2, unit_parse_strict="silent")
        else:
            gti = None

        if method == "lc":
            lcins = get_lightcurve_obj(tb, gti)
        elif method == "tb":
            lcins = (tb, gti)
        return lcins
    elif isinstance(lc, np.ndarray):
        lcins = Lightcurve(time=lc[:, 0], counts=lc[:, 1])
        return lcins
    elif isinstance(lc, Lightcurve):
        lcins = lc
        return lcins
    else:
        raise Exception("Not support current input")


def open_gti(gti, ndarray=False):
    """
    用于打开gti文件，返回dataframe对象

    """
    if isinstance(gti, str):  # 文件名情况，总是使用astropy table
        # 判断hdu数量
        with fits.open(gti) as hdul:
            lens = len(hdul)

        if lens >= 2:
            gti = Table.read(gti, 1, unit_parse_strict="silent")
        else:
            gti = None

        if ndarray:
            if gti is not None:
                return gti.to_pandas().values
            else:
                return np.array([])
        else:
            return gti.to_pandas()
    elif isinstance(gti, np.ndarray):
        if ndarray:
            return gti
        else:
            return pd.DataFrame(gti, columns=["TSTART", "TSTOP"])
    elif isinstance(gti, pd.DataFrame):
        if ndarray:
            return gti.to_numpy()
        else:
            return gti
    else:
        raise Exception("Not support current input")


def get_lightcurve_obj(lctb: Table, lcgti=None, header=None):
    """
    将astropy table转为Lightcurve对象
    """
    if lcgti:
        gti = lcgti.to_pandas().values
    else:
        gti = extract_gti_from_lc(lctb.columns[0].value)

    lcobj = Lightcurve(
        time=lctb.columns[0].value,
        counts=lctb.columns[1].value,
        err=lctb.columns[2].value,
        gti=gti,
        skip_checks=True,
    )

    if header:
        lcobj.mjdref = header["MJDREFI"] + header["MJDREFF"]
        lcobj.mission = header["TELESCOP"]
        lcobj.instr = header["INSTRUME"]

    return lcobj


def get_gti_from_lc(lc):
    """
    获取GTI Table
    """
    lcins = open_lc(lc)
    return np.array(lcins.gti)


def extract_gti_from_lc(lc, threshold=None):
    """
    从光变曲线中提取GTI；如果没有提供阈值，则用分箱和最小间隔的均值作为阈值
    """
    times = np.array(lc)
    diffs = np.diff(times)

    if not threshold:
        binsize = np.min(diffs)
        NGTI_lens = diffs[diffs != binsize]
        if NGTI_lens.size == 0:
            threshold = binsize  # 如果是连续的光变曲线
        else:
            threshold = np.mean([min(NGTI_lens), binsize])

    # where找到NGTI的在diffs中的索引，
    # 对应在times中就是NGTI开始前的GTI最后一个时间点的索引，
    # 因此gaps中数据就是每段GTI最后一个时间点的索引
    gaps = np.where(diffs > threshold)[0]

    gti = []
    start = 0  # 双指针
    for end in gaps:
        gti.append([times[start], times[end]])
        start = end + 1
    gti.append([times[start], times[-1]])
    return np.array(gti)


def get_exposure_from_gti(gti):
    """
    计算总曝光时间
    """
    if isinstance(gti, np.ndarray):
        if gti.size == 0:
            return 0
        try:
            if gti.shape[1] == 2:
                pass
        except IndexError:
            raise Exception("lc array must be 2D.")
        return np.sum(gti[:, 1] - gti[:, 0])
    elif isinstance(gti, list):
        get_exposure_from_gti(np.array(gti))


def get_common_GTI(*gtiarray, endpoints=False):
    """
    输入多个GTI数组，返回交集的GTI ndarray
    """
    gti_set = None  # 初始化为空

    for gtiarr in gtiarray:
        # 如果输入为 numpy ndarray 或 pandas DataFrame，则转为列表
        if isinstance(gtiarr, list):
            pass
        elif isinstance(gtiarr, np.ndarray):
            gtiarr = gtiarr.tolist()
        elif isinstance(gtiarr, pd.DataFrame):
            gtiarr = gtiarr.values.tolist()
        elif isinstance(gtiarr, None):
            continue

        # 先处理单次GTI
        now_gti_set = None
        for start, end in gtiarr:
            current_interval = P.closed(start, end)

            # 第一次直接赋值 gti_set
            if now_gti_set is None:
                now_gti_set = current_interval
            else:
                # 使用并组合断续的一次GTI
                now_gti_set |= current_interval

        # 取多次GTI的交集
        if gti_set is None:
            gti_set = now_gti_set
        else:
            if now_gti_set is None:
                continue
            else:
                gti_set &= now_gti_set

    # 如果没有交集，返回空数组
    if gti_set is None or gti_set.empty:
        return np.array([])

    # 提取区间的起止点
    gti_rawlist = P.to_data(gti_set)
    start_list = [i[1] for i in gti_rawlist]
    stop_list = [i[2] for i in gti_rawlist]
    gti_set_arr = np.array(list(zip(start_list, stop_list)))

    # 根据 endpoints 参数判断是否要过滤掉相等的起止点
    if not endpoints and gti_set_arr.size != 0:
        gti_set_arr = gti_set_arr[gti_set_arr[:, 0] != gti_set_arr[:, 1]]

    return gti_set_arr


def get_union_gti(*gtiarray):
    """
    输入多个GTI数组，返回并集的GTI ndarray
    """
    gti_set = None  # 初始化为空

    for gtiarr in gtiarray:
        # 如果输入为 numpy ndarray 或 pandas DataFrame，则转为列表
        if isinstance(gtiarr, list):
            pass
        elif isinstance(gtiarr, np.ndarray):
            gtiarr = gtiarr.tolist()
        elif isinstance(gtiarr, pd.DataFrame):
            gtiarr = gtiarr.values.tolist()
        elif isinstance(gtiarr, None):
            continue

        # 处理单次GTI
        now_gti_set = None
        for start, end in gtiarr:
            current_interval = P.closed(start, end)

            # 第一次直接赋值 now_gti_set
            if now_gti_set is None:
                now_gti_set = current_interval
            else:
                # 使用并组合当前的GTI
                now_gti_set |= current_interval  # 合并区间

        # 取多次GTI的并集
        if gti_set is None:
            gti_set = now_gti_set
        else:
            if now_gti_set is None:
                continue
            else:
                gti_set |= now_gti_set  # 并集运算

    # 如果没有并集，返回空数组
    if gti_set is None or gti_set.empty:
        return np.array([])

    # 提取区间的起止点
    gti_rawlist = P.to_data(gti_set)
    start_list = [i[1] for i in gti_rawlist]
    stop_list = [i[2] for i in gti_rawlist]
    gti_set_arr = np.array(list(zip(start_list, stop_list)))

    return gti_set_arr


def get_timerange_from_lc(lc, mjd=False):
    """
    获取光变曲线的时间覆盖范围
    """
    lcins = open_lc(lc)
    time_start = lcins.time[0]
    time_stop = lcins.time[-1]
    return {"tstart": time_start, "tstop": time_stop}


def get_meanrate_from_lc(lc):
    """
    获取平均计数率
    """
    lcins = open_lc(lc)
    return lcins.meanrate


def matching_time_unit(time):
    """
    时间单位匹配，1s，1ms，1us
    """
    if not isinstance(time, (int, float)):
        try:
            time = float(time)
        except ValueError:
            raise ValueError(
                f"matching_time_unit: Invalid input, cannot convert {time} to float"
            )

    if time >= 1.0:
        return f"{round(time)}sec"
    elif time < 1.0 and time > 1e-3:
        return f"{round(time*1e3)}ms"
    elif time < 1e-3:
        return f"{round(time*1e6)}us"


def get_timeobj(TT, fitsfile=None):
    """
    输入HXMT的TT时间，返回时间对象
    """
    TT = float(TT)
    if fitsfile:
        hdul = fits.open(fitsfile)
        start_hxmt = Time(
            hdul[1].header["MJDREFI"] + hdul[1].header["MJDREFF"], format="mjd"
        )
        unixTime = start_hxmt.unix + TT
        return Time(unixTime, format="unix")
    else:
        start_hxmt = Time(55927.00076601852, format="mjd")
        unixTime = start_hxmt.unix + TT
        return Time(unixTime, format="unix")


def convert_TT2MJD(TT, fitsfile=None):
    """
    将HXMT的TT时间转换为MJD
    """
    return get_timeobj(TT, fitsfile).mjd


def convert_TT2UTC(TT, fitsfile=None):
    """
    将HXMT的TT时间转换为UTC时间
    """
    return get_timeobj(TT, fitsfile).utc.iso.split(".")[0].replace(" ", "T")


def get_sliced_lc(lc, sliced_num):
    """
    如果lcobj有多段GTI, 返回序号为sliced_num的光变曲线
    """
    if not isinstance(lc, Lightcurve):
        raise Exception("Not support current input")

    nowgti = lc.gti[sliced_num]
    return lc.truncate(nowgti[0], nowgti[1], method="time")


def slice_multiple_gtis(gti_lists, slice_length=128, index=False):
    """
    将多个GTI列表进行切片,支持Table、nparray，并对结果进行排序；当index=True时，返回行号和切片结果
    """
    sliced_gtis = []
    for gti in gti_lists:
        start, end = gti
        while start + slice_length <= end:
            sliced_gtis.append([start, start + slice_length])
            start += slice_length
    # 对sliced_gtis进行排序
    sliced_gtis.sort(key=lambda x: x[0])

    sliced_gtis = np.array(sliced_gtis)

    if index and len(sliced_gtis) > 0:
        row_numbers = np.arange(1, sliced_gtis.shape[0] + 1).reshape(-1, 1)
        return np.hstack((row_numbers, sliced_gtis))

    return sliced_gtis


def slice_lc_by_gti(lc, sliced_gtis):
    """
    按指定的GTI数组将Lightcurve对象切片, 返回Lightcurve列表
    """
    sliced_lcs = []
    for gti in sliced_gtis:
        sliced_lcs.append(lc.truncate(gti[0], gti[1], method="time"))
    return sliced_lcs


def slice_lc_segment_from_lc(lc, segment):
    """
    将光变曲线切片
    """
    lcins = open_lc(lc)
    lcins_segment = lcins.truncate(segment[0], segment[1], method="time")
    return lcins_segment


def concat_lc(*lightcurve):
    """
    组合光变曲线，返回组合后的Lightcurve对象
    """
    lightcurves = lightcurve
    concated = None
    for i, lc in enumerate(lightcurves):
        if isinstance(lc, Lightcurve):  # 如果是Lightcurve对象
            nowlc = lc
        elif isinstance(lc, str):  # 如果是文件路径
            try:
                nowtb = Table.read(lc, 1, unit_parse_strict="silent")
                nowtb_gti = Table.read(lc, 2, unit_parse_strict="silent")
                nowlc = get_lightcurve_obj(nowtb, nowtb_gti)
                print(f"the exposure of light curve {i} is {nowlc.tseg}")
            except:
                Exception(f"Could not load the lightcurve file:\n\t{lc}")
        else:
            Exception("Not support current input")

        if not concated:
            concated = nowlc
        else:
            concated = concated.join(nowlc)
    print(f"The time coverage of concated lc is {concated.tseg} s")
    return concated


def generate_netlc(lcraw, lcbkg):
    """
    生成净光变曲线，返回hdul对象
    """
    rawlc_hdul = fits.open(lcraw)
    rawlc_binsize = rawlc_hdul[1].header["TIMEDEL"]

    rawlc_tb, _ = open_lc(lcraw, method="tb")
    bkglc_tb, _ = open_lc(lcbkg, method="tb")

    bkglc_tb_rate = bkglc_tb.columns[1]  # 默认计数率在第二列
    bkglc_tb_err = bkglc_tb.columns[2]  # 默认计数率误差在第三列

    # 先获取列名，再进行修改
    rawlc_tb_counts_index = rawlc_tb.colnames[1]
    rawlc_tb_err_index = rawlc_tb.colnames[2]
    rawlc_tb[rawlc_tb_counts_index] -= bkglc_tb_rate * rawlc_binsize
    rawlc_tb[rawlc_tb_err_index] = (
        rawlc_tb.columns[2] ** 2 + (bkglc_tb_err * rawlc_binsize) ** 2
    ) ** 0.5

    rawlc_hdul[1] = fits.BinTableHDU(
        rawlc_tb,
        header=rawlc_hdul[1].header,
    )
    return rawlc_hdul


def convert_lc2fits(lightcurve, template_fits, fitsname, headers=None):
    hdul = fits.open(template_fits)
    lcdf = lightcurve.to_pandas()[["time", "counts", "counts_err"]]
    lcdf.columns = ["TIME", "COUNTS", "ERROR"]
    lctb = Table.from_pandas(lcdf)
    if headers is None:
        new_header = hdul[1].header
    else:
        new_header = headers
    hdul[1] = fits.BinTableHDU(lctb, header=new_header)
    gtiarr = lightcurve.gti
    gtidf = pd.DataFrame(gtiarr, columns=["TSTART", "TSTOP"])
    gtitb = Table.from_pandas(gtidf)
    gtiBTH = fits.BinTableHDU(gtitb)
    hdul[2].data = gtiBTH.data
    hdul.writeto(f"{fitsname}", overwrite=True)
    return hdul


def plotlc_concat(
    *lcfiles, col_names=None, binsize=1.0, extra_marker_range=None, **kwargs
):
    if col_names is None:
        col_names = ["TIME", "COUNTS", "ERROR"]

    combined_gti = None
    lc_ins_lst = []
    if extra_marker_range is None:
        colors = ["#FA9D3A", "#3369a3"] * (len(lcfiles) // 2 + 1)
    else:
        colors = ["#FA9D3A", "#3369a3"] * (len(extra_marker_range) // 2 + 1)

    for lcfile in lcfiles:
        lc_ins = open_lc(lcfile)
        lc_ins = lc_ins.rebin(dt_new=binsize)
        if len(lc_ins_lst) == 0:
            lc_combined = lc_ins
            lc_ins_lst.append(lc_combined)
            combined_gti = lc_ins.gti
        else:
            lc_ins_lst.append(lc_ins)
            lc_combined = concat_lc(lc_combined, lc_ins)
            combined_gti = np.concatenate((combined_gti, lc_ins.gti))

    fig = plt.figure(figsize=(16 + 3 * (len(lcfiles) - 1), 6))

    # 确定x轴范围
    xlims = [(interval - lc_combined.time[0]).tolist() for interval in combined_gti]
    bax = brokenaxes(xlims=xlims, hspace=0.01, **kwargs)
    for i, lc in enumerate(lc_ins_lst):
        if i == 0:
            label = f"binsize={binsize:.5g}s"
        else:
            label = None
        bax.plot(
            lc.time - (lc_combined.time[0]),
            lc.counts,
            color=colors[i],
            label=label,
        )
    bax.axhline(y=lc_combined.meancounts, linestyle="-", color="#114486a1")
    bax.plot(
        [],
        [],
        " ",
        label=rf"$C_{{avg}}$/$\sigma_{{C}}$={lc_combined.meancounts:.5g}/{np.std(lc_combined.counts):.5g}",
    )
    bax.plot(
        [], [], " ", label=rf"$\Delta t$={get_exposure_from_gti(lc_combined.gti):.2f}"
    )
    bax.set_xlabel("Time (s, from %d)" % (lc_combined.time[0]))
    bax.set_ylabel(f"{col_names[1]}")
    bax.legend(loc="best")
    if isinstance(lcfile, str):
        bax.set_title("\n".join(lcfiles))

    # 如果需要额外的高亮区域显示
    if not extra_marker_range is None:
        for i, (start, stop) in enumerate(extra_marker_range):
            start = start - (lc_combined.time[0])
            stop = stop - (lc_combined.time[0])
            print(start, stop)
            bax.axvspan(
                start,
                stop,
                alpha=0.3,
                color=colors[i],
                # ymin=0.45,
                # ymax=0.55,
            )

    return fig, bax


def plotlc_stack(
    *lcfiles,
    binsize=1.0,
    gti_init=None,
    col_names=None,
    break_yaxis=True,
    figsize=None,
    legend=True,
    plotmode: Literal["plot", "errorbar"] = "plot",
):
    if col_names is None:
        col_names = ["TIME", "COUNTS", "ERROR"]

    # 获取GTI
    lclst = []
    gti = None
    gtilst = []
    for index, lcfile in enumerate(lcfiles):
        if gti_init:  # 如果已经指定GTI
            gti = open_gti(gti_init, ndarray=True)
            break

        lc = open_lc(lcfile)
        if len(lc.time) == 0 or len(lc.counts) == 0:
            raise ValueError(
                f"Warning: No data in {index}:{os.path.basename(lcfile)}, skipping."
            )
        gtilst.append(lc.gti)

        lc = lc.rebin(dt_new=binsize)
        lclst.append(lc)

    if gti is None:
        gti = get_union_gti(*gtilst)  # 返回GTI的并集，以显示所有数据
        if gti.shape[0] == 0:
            raise ValueError("No valid GTI found.")

    # 如果允许断开yaxis
    ylims = None
    if break_yaxis:
        for lc in lclst:
            counts = lc.counts
            counts_err = lc.counts_err
            if ylims is None:
                ylims = P.closed(
                    np.min(counts) - 1.5 * counts_err[np.argmin(counts)],
                    np.max(counts) + 1.5 * counts_err[np.argmax(counts)],
                )
            else:
                ylims |= P.closed(
                    np.min(counts) - 1.5 * counts_err[np.argmin(counts)],
                    np.max(counts) + 1.5 * counts_err[np.argmax(counts)],
                )

        ylims = P.to_data(ylims)
        start_list = [i[1] for i in ylims]
        stop_list = [i[2] for i in ylims]
        ylims = list(zip(start_list, stop_list))

    # 绘图
    bax = None
    for lc in lclst:
        if bax is None:
            # 堆叠模式在这里创建axes
            if figsize is None:
                figsize = (16, 6 + len(lclst) * 2)
            fig = plt.figure(figsize=figsize)
            bax = brokenaxes(xlims=(gti - gti[0, 0]).tolist(), ylims=ylims, hspace=0.01)
        if plotmode == "plot":
            bax.plot(
                lc.time - (gti[0, 0]),
                lc.counts,
                label=rf"$C_{{avg}}$/$\sigma_{{C}}$={lc.meancounts:.5g}/{np.std(lc.counts):.5g}",
            )
        elif plotmode == "errorbar":
            bax.errorbar(
                x=lc.time - (gti[0, 0]),
                y=lc.counts,
                yerr=lc.counts_err,
                label=rf"$C_{{avg}}$/$\sigma_{{C}}$={lc.meancounts:.5g}/{np.std(lc.counts):.5g}",
                linestyle="",
                marker=".",
            )

    # bax.axhline(y=lc.meancounts, linestyle="-", color="#114486a1")
    bax.grid(axis="both", which="major", ls="-")
    bax.grid(axis="both", which="minor", ls="--", alpha=0.4)
    bax.set_xlabel("Time (s, from %d)" % (lc.time[0]))
    bax.set_ylabel(f"{col_names[1]}")
    bax.plot(
        [],
        [],
        " ",
        label=f"binsize={binsize:.5g}s",
    )
    bax.plot([], [], " ", label=rf"$\Delta t$={get_exposure_from_gti(gti):.2f}")
    if legend:
        bax.legend(loc="best")
    if isinstance(lcfile, str):
        bax.set_title("\n".join(lcfiles))

    return fig, bax


# 目前只支持绘制time/counts时间序列，不支持绘制times/rate
def plotlc(*lcfiles, mode="stack", **kwargs):
    if mode == "concat":
        return plotlc_concat(*lcfiles, **kwargs)
    elif mode == "stack":
        return plotlc_stack(*lcfiles, **kwargs)


def plotbkg(bkgfile, col_names=None, binsize=1.0):
    if col_names is None:
        header = fits.getheader(bkgfile, 1)
        if "HE" in header["INSTRUME"]:
            col_names = ["Time", "Rate", "Error"]
        elif "ME" in header["INSTRUME"]:
            col_names = ["Time", "RATE", "Error"]
        elif "LE" in header["INSTRUME"]:
            col_names = ["Time", "RATE", "Error"]
        else:
            col_names = ["Time", "RATE", "ERROR"]

    lcbkg = open_lc(bkgfile)
    # HXMT背景文件实际记录的是计数率
    lcbkg.counts = lcbkg.counts * lcbkg.dt

    fig = plt.figure(figsize=(16, 8))

    lcbkg = lcbkg.rebin(dt_new=binsize)
    times = lcbkg.time
    counts = lcbkg.counts
    gti = lcbkg.gti

    bax = brokenaxes(xlims=(gti - times[0]).tolist(), hspace=0.02)
    bax.plot(times - (times[0]), counts, label=f"binsize={binsize:.5g} s")
    bax.grid(axis="both", which="major", ls="-")
    bax.grid(axis="both", which="minor", ls="--", alpha=0.4)
    bax.set_xlabel("Time (s, from %d)" % (times[0]))
    bax.set_ylabel(f"COUNTS")

    bax.axhline(
        y=lcbkg.meancounts,
        linestyle="-",
        color="#114486",
    )

    bax.plot(
        [],
        [],
        " ",
        label=rf"$C_{{avg}}$/$\sigma_{{C}}$={lcbkg.meancounts:.5g}/{np.std(lcbkg.counts):.5g}",
    )
    bax.plot([], [], " ", label=rf"$\Delta t$={get_exposure_from_gti(lcbkg.gti):.2f}")
    bax.legend(loc="best")
    bax.set_title(bkgfile)
    return fig, bax
