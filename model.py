# Precession model
import numpy as np
import os
from math import sin, cos

home = os.getenv("HOME")

mdots = np.array([2, 7, 12, 24])

PATH_TO_SIMUS = f"{home}/Documents/papers/qpes/simulations_output/cutoff"

print(f"Simulations path: {PATH_TO_SIMUS}")

FILE = f"{PATH_TO_SIMUS}/Lx_Mdot"

mdot_files = [np.genfromtxt(FILE + "%d.txt" % mdot, names=["i_rad", "flux"]) for mdot in mdots]

TO_RAD = np.pi / 180
two_pi = 2 * np.pi

def get_mdot_file(input_mdot):
    """Casts input mdot into one of the available mdot values and retrieves the corresponding values"""
    return mdot_files[np.argmin(np.abs(input_mdot - mdots))]


def harm_model_rebinned(params, times, exposures, rebin_factor=10):
    """Construct precessing model here
    Parameters
    ----------
    params[0]:period
    params[1]:phi (phase)
    params[2]:i_0
    params[3]: di
    params[4]: mdot
    params[5]: A

    """
    # make a grid N times smaller
    bin_starts = times - exposures / 2
    bin_ends = times + exposures / 2
    finer_times = np.linspace(times.min(), times.max(), len(times) * rebin_factor)

    model = harm_model(params, finer_times)

    rebinned_model = []
    for start, end in zip(bin_starts, bin_ends):
        mask = (finer_times >= start) & (finer_times <= end)  # Points in the current bin
        if np.any(mask):  # If there are points in this bin
            rebinned_model.append(np.mean(model[mask]))
    return np.array(rebinned_model)

    dt = np.median(np.diff(times))

    lc = lc.bin(time_bin_start = times - dt, time_bin_end=times + dt)

    new_shape = finer_times.shape[0] //  rebin_factor
    shape = (new_shape, model.shape[0] // new_shape)
    return lc.flux
    return np.mean(model.reshape(shape), axis=-1)



def harm_model(params, times):
    """Construct precessing model here
    inputparams
    params[0]:period
    params[1]:phi (phase)
    params[2]:i_0
    params[3]: di
    params[4]: A

    """
    period, phase_0, incl, dincl, mdot, A = params  # mdot

    incl_rad = incl * TO_RAD
    dincl_rad = dincl * TO_RAD

    if incl < 0.0:
        raise ValueError("Incl must be positively defined! (i = %.2f)" % incl)
    # calculate the phase
    phase = two_pi * (phase_0 + times / period)
    ##wrapped_phase = phase % (2 * np.pi) # between 0 and 2pi --> No need as numpy handles that correctly
    # CONE
    # calculate cos i based on Eq 61 from Abolmasov+2009
    cosi = cos(dincl_rad) * cos(incl_rad) \
                + sin(dincl_rad) * sin(incl_rad) * np.cos(phase)

    i_t = np.arccos(cosi)

    # if i>90, we see the undercone
    #i_t = np.where(i_t > 90, i_t - 90, i_t)
    i_t_undercone = np.abs(np.pi - np.abs(i_t))

    flux_i = get_mdot_file(mdot)

    #flux_i = np.genfromtxt(FILE + "%d.txt" % casted_mdot, names=["i_rad", "flux"])
    #i_t_undercone = np.where(i_t_undercone > 90, i_t_undercone - 90, i_t_undercone)
    # find the fluxes at the calculate inclinations
    #i_file = np.searchsorted(flux_i["i_rad"], i_t) - 1
    #model_rates =  flux_i["flux"][i_file]#np.interp(np.abs(i_t), flux_i["i_rad"], flux_i["flux"])
    model_rates = np.interp(np.abs(i_t), flux_i["i_rad"], flux_i["flux"], right=0)
    #i_angles = [np.argmin(np.abs(i - flux_i["i_rad"])) for i in np.abs(i_t)]
    #model_rates = flux_i["flux"][i_angles]
    #i_file_undercone = np.searchsorted(flux_i["i_rad"], i_t_undercone) - 1
    #undercone =  flux_i["flux"][i_file_undercone]
    undercone = np.interp(np.abs(i_t_undercone), flux_i["i_rad"], flux_i["flux"], right=0)

    return model_rates + A + undercone