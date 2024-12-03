import os

import numpy as np
from astropy.table import Table
from astropy.modeling import Fittable1DModel, Parameter
from stingray import EventList, Powerspectrum
from stingray.modeling import PSDParEst
from stingray.modeling.parameterestimation import PSDLogLikelihood

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

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


class CustomPowerLawPlusConst(Fittable1DModel):
    # Define parameters
    amplitude = Parameter(default=1.0)
    x_0 = Parameter(default=1.0)  # reference frequency
    alpha = Parameter(default=1.0)  # power-law index

    # Fixed constant
    const = 2.0

    @staticmethod
    def evaluate(x, amplitude, x_0, alpha):
        """Evaluate the PowerLaw + Constant model."""
        powerlaw = amplitude * (x / x_0) ** (-alpha)
        return powerlaw + CustomPowerLawPlusConst.const

    @staticmethod
    def fit_deriv(x, amplitude, x_0, alpha):
        """Derivative of the model with respect to parameters."""
        d_amplitude = (x / x_0) ** (-alpha)
        d_x_0 = -amplitude * alpha * (x / x_0) ** (-alpha - 1) * (-1 / x_0)
        d_alpha = -amplitude * (x / x_0) ** (-alpha) * np.log(x / x_0)
        return [d_amplitude, d_x_0, d_alpha]


def open_evt(evtfile):
    tb = Table.read(evtfile, hdu=1, unit_parse_strict="silent")
    evt = EventList(
        time=tb["Time"],
    )
    return evt


def modeling_evtpds(pdsobj):
    pass


def plotpds_from_evt(evtfile, fit=True, rebin=-0.003, dt=1 / 128):
    """ """
    evt = open_evt(evtfile)
    obj = Powerspectrum.from_events(evt, dt=dt, norm="leahy")

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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.step(x=obj.freq, y=obj.power, where="mid")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(evtfile)

    x_lower = obj.freq[0] / 2
    x_upper = obj.freq[-1]
    ax.set_xlim(x_lower, x_upper)

    if not fit:
        return fig, ax

    # Fit only freq <= 0.1
    mask_range = 0.1
    fit_mask = obj.freq <= mask_range

    # Create a new Powerspectrum object for fitting
    obj_fit = obj.apply_mask(fit_mask)

    # Fit model
    fit_model = CustomPowerLawPlusConst(amplitude=1.0, x_0=1.0, alpha=1.0)

    parest = PSDParEst(obj_fit, fitmethod="L-BFGS-B", max_post=False)
    loglike = PSDLogLikelihood(obj_fit.freq, obj_fit.power, fit_model, m=obj_fit.m)

    res = parest.fit(loglike, fit_model.parameters)

    fitmod = res.model

    # Plot fitted models over the full frequency range
    ax.plot(
        obj.freq,
        fitmod(obj.freq),
        color="#41699a",
        label=rf"$\Gamma=${res.p_opt[-1]:.2f}",
    )
    ax.legend()

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(1, np.max(obj.power))
    return fig, ax
