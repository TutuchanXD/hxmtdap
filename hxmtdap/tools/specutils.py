# 一些能谱工具
import re

import numpy as np
from scipy.integrate import quad
from astropy.io import fits


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


def standard_data_xcm1(expID):
    xcm_data = (
        "statistic chi\n"
        + "cd ../output\n"
        + f"data 1:1 {expID}_LE.pha\n"
        + "\n"
        + f"data 2:2 {expID}_ME.pha\n"
        + "\n"
        + f"data 3:3 {expID}_HE.pha\n"
        + "\n"
        + "cd ../fit\n"
        + "ignore 1:**-3.0 10.0-** 2:**-10.0 30.0-** 3:**-30.0 150.0-**\n"
    )
    return xcm_data


def standard_mcmc(mcmcfilename):
    mcmc_string = (
        "fit\n"
        + "energies 0.01 1000. 2000 log\n"
        + "chain len 50000\n"
        + "chain burn 50000\n"
        + "chain walker 200\n"
        + f"chain run {mcmcfilename}\n"
    )
    return mcmc_string
