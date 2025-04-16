# 一些能谱工具
import re

import numpy as np
from scipy.integrate import quad
from astropy.io import fits
from astropy.time import Time
import os


def update_grp_file_paths(phafile):
    """
    Updates a PHA file by changing the absolute paths in 'respfile' and 'backfile'
    header keys to just the file basenames.

    Parameters
    ----------
    phafile : str
        Path to the PHA file
    """
    with fits.open(phafile, mode="update") as hdul:
        header = hdul[1].header

        # Get current values and update to basenames if they exist
        if "respfile" in header:
            header["respfile"] = os.path.basename(header["respfile"])

        if "backfile" in header:
            header["backfile"] = os.path.basename(header["backfile"])

        # Changes are automatically saved when the file is closed


def get_exposure_from_pha(phafile):
    """
    从PHA文件中提取曝光时间
    """
    with fits.open(phafile) as hdulist:
        exposure = hdulist[1].header["EXPOSURE"]
    return exposure


def get_mjdtime_from_pha(phafile):
    """
    从PHA文件中提取MJD时间
    """
    with fits.open(phafile) as hdulist:
        dstart = hdulist[1].header["DATE-OBS"]
        dstop = hdulist[1].header["DATE-END"]

    mstart = Time(dstart).mjd
    mstop = Time(dstop).mjd
    mjdtime = (mstart + mstop) / 2
    mjderr = (mstop - mstart) / 2
    return mjdtime, mjderr


def power_law_model(E, K, alpha):
    return K * E ** (-alpha)


def calculate_integral(E1, E2, K, alpha):
    return quad(power_law_model, E1, E2, args=(K, alpha))[0]


def extract_model_results_from_xcm(xcmfilepath):
    with open(xcmfilepath, "r") as xcm:
        content = xcm.read()

    if "model" not in content:
        raise ValueError("The model results is not found in {}".format(xcmfilepath))

    pattern = r"(method.*?bayes off)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("The model results is not found in {}".format(xcmfilepath))

    return match.group(1) + "\n"


def generate_GTI(time_array):
    """
    根据Array生成一个GTI文件，主要用于HXMTDAS背景生成时与谱的GTI同步
    """
    timearr = np.array(time_array)
    start_arr = timearr[:, 0]
    stop_arr = timearr[:, 1]

    start_col = fits.Column(name="START", format="f8", array=start_arr)
    stop_col = fits.Column(name="STOP", format="f8", array=stop_arr)
    gti_col = fits.ColDefs([start_col, stop_col])

    primary_hdu = fits.PrimaryHDU()
    secondary_hdu = fits.BinTableHDU.from_columns(gti_col)

    hdulist = fits.HDUList([primary_hdu, secondary_hdu])
    return hdulist
