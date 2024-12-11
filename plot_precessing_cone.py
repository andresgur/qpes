#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   plot_precessing_cone.py
@Time    :   2024/12/11 12:16:53
@Author  :   Andrés Gúrpide Lasheras
@Contact :   a.gurpide-lasheras@soton.ac.uk
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from model import two_pi, TO_RAD, PATH_TO_SIMUS


def harm_model(times, period, phase_0, incl, dincl):
    """Construct precessing model here
    inputparams
    params[0]:period
    params[1]:phi (phase)
    params[2]:i_0
    params[3]: di
    params[4]: mdot
    """
    print("Incl: %.1f" % incl)
    print("Delta Incl: %.1f" % dincl)

    if incl < 0.0:
        raise ValueError("Incl must be positively defined! (i = %.2f)" % incl)
    
    incl_rad = incl * TO_RAD
    dincl_rad = dincl * TO_RAD
    # calculate the phase
    phase = two_pi * (phase_0 + times / period)
    # CONE
    # calculate cos i based on Eq 61 from Abolmasov+2009
    cosi = np.cos(dincl_rad) * np.cos(incl_rad) \
                + np.sin(dincl_rad) * np.sin(incl_rad) * np.cos(phase)
    # convert to i, this works as long as i<180. If i SHOULD be negative, it is mirrored
    #sini = np.sqrt(1 - cosi**2)
    #i_t = np.arctan2(sini, cosi)
    # this should be ok since the model is symmetric around i=0
    i_t = np.arccos(cosi)#np.where(cosi > 0, np.arccos(cosi) , np.arccos(cosi)-np.pi)#


    print("Min angle: %.2f Max angle: %.2f" %(i_t[np.argmin(i_t)] / TO_RAD, i_t[np.argmax(i_t)] / TO_RAD))
    #print(i_t_undercone[np.argmin(i_t_undercone)], i_t_undercone[np.argmax(i_t_undercone)])

    # if i>90, we see the undercone
    i_t_undercone = np.pi - np.abs(i_t)
    #i_t_undercone = np.where(i_t_undercone > 90, i_t_undercone - 90, i_t_undercone)
    # find the fluxes at the calculate inclinations
    model_rates = np.interp(np.abs(i_t), flux_i["i"], flux_i["flux"], right=0)

    undercone = np.interp(np.abs(i_t_undercone), flux_i["i"], flux_i["flux"], right=0)

    return  model_rates, undercone, i_t, i_t_undercone, phase # + undercone


if os.path.isfile("/home/andresgur/.config/matplotlib/stylelib/paper.mplstyle"):
    plt.style.use("/home/andresgur/.config/matplotlib/stylelib/paper.mplstyle")


ap = argparse.ArgumentParser(description='Plot the flux of a precesing cone as a function of time and phase')
ap.add_argument("-m", "--mdot", nargs="?", default=2, choices=[2, 7, 12, 24], help="Mdot in Eddington units", type=float)
ap.add_argument("-A", "--A", nargs="?", default=0.1, help="Quiescent constant level (in Edd units). Default: 0", type=float)
ap.add_argument("-o", "--outdir", nargs="?", default="cones", help="Output dir", type=str)
ap.add_argument("-p", "--period", nargs="?", default=8743.0, help="Precessing period in seconds", type=float)
ap.add_argument("-di", "--dincl", nargs="+", default=[5.], help="Amplitude of the inclination during the precessing cycle in degrees. Default 5",
                type=float)
ap.add_argument("-i", "--incl", nargs="?", default=10, help="Inclination of the cone in degrees. Default 10",
                type=float)
args = ap.parse_args()
outdir = args.outdir
if not os.path.isdir(outdir):
    os.mkdir(outdir)


print("Plotting the interpolator")
# change this to your Lx_Mdot folder where both mdot files are stored
plt.figure()
angles = np.linspace(0, 90, 100)
mdot = args.mdot
FILE = f"{PATH_TO_SIMUS}/" + "Lx_Mdot%d.txt" % mdot
flux_i = np.genfromtxt(FILE, names=["i", "flux"])
model_rates = np.interp(angles *TO_RAD, flux_i["i"], flux_i["flux"])
fig = plt.figure(figsize=(16, 8))
plt.plot(angles, model_rates, label="Interpolated", color="C1")
plt.scatter(flux_i["i"] / TO_RAD, flux_i["flux"], zorder=-10)
plt.legend()
plt.yscale("linear")
plt.xlabel("$i$ ($^\circ$)")
plt.ylabel("$L$ [0.3 $-$ 10 keV] (Edd)")
plt.axvline(args.incl, ls="--", color="black")
plt.axvspan(args.incl - args.dincl[0], args.incl + args.dincl[0], alpha=.2)
plt.yscale("log")
plt.savefig("%s/interpolated_m%d.png" % (outdir, mdot),
            facecolor="white", bbox_inches="tight")
plt.close(fig)

incl = args.incl
phase_0 = 0.1
period = args.period
A = args.A
times = np.arange(0, period * 2, period / 10000)
for dincl in args.dincl:


    rate, underate, angles, angles_down, phase_t = harm_model(times, period=period, phase_0=phase_0, incl=incl,
                         dincl=dincl, rescale=False)
    print("Cone peak luminosity (Edd):%.3f" % np.max(rate + underate))

    fig, axes = plt.subplots(2,1, figsize=(16, 8), sharex=True, gridspec_kw={"hspace":0})
    cone_lines = axes[0].plot(times, angles / TO_RAD, color="C0", label="Cone") #  - times[0]
    undercone_lines = axes[0].plot(times, angles_down / TO_RAD, color="C1", label="Undercone") # - times[0]
    axes[0].set_title("$\dot{m}_\\text{Edd} = %d$, $i = %.1f^\circ$ $\Delta i$ = %.1f$^\circ$ A = %.1f Edd" % (mdot, incl, dincl, A))
    axes[0].set_ylabel("Precession angle ($^\\circ$)")
    axes[0].axhline(incl, ls="--", color="black")
    axes[0].invert_yaxis()
    axes[0].legend()

    if True:
        twin_ax = axes[0].twiny()
        twin_ax.set_ylabel("Phase")
        xticks = axes[0].get_xticks()
        x2labels = ["%.2f" %  (phase_t[np.argmin(np.abs(times - xtick))] / two_pi) for xtick in xticks] #  - times[0
        twin_ax.set_xticks(xticks)
        twin_ax.set_xticklabels(x2labels)
        twin_ax.set_xlim(axes[0].get_xlim())
        twin_ax.set_xlabel("Phase")

    #twin_ax.plot(phase_t % (2 * np.pi), rate + underate + A, color="black", alpha=0)
    #plt.figure()
    #plt.plot(times - times[0], cosi, label="$i = %d$, $\Delta i = %d$" %(incl, dincl))#label="$\dot{m} = %d$" % mdot)
    if incl + dincl > 90:
        print("Plotting undercone")
        axes[1].plot(times, underate + A, ls="--", color=undercone_lines[0].get_color())#label="$\dot{m} = %d$" % mdot)

    axes[1].plot(times, rate + A, ls=":", color=cone_lines[0].get_color())#label="$\dot{m} = %d$" % mdot)
    axes[1].plot(times, rate + underate + A, color="black", label="Total", alpha=0.5, zorder=-10)#label="$\dot{m} = %d$" % mdot)
    #if False:
    #    axes[1].axvline(times[np.argmin(np.abs(phase_t -1))] - times[0], ls="--", ymin=0.7, color="black")
    #    axes[1].axvline(times[np.argmin(np.abs(phase_t -2))] - times[0], ls="--", ymin=0.7, color="black")
    #    axes[1].axvline(times[np.argmin(np.abs(phase_t -3))] - times[0], ls="--", ymin=0.7, color="black")
    #    axes[1].axvline(times[np.argmin(np.abs(phase_t -4))] - times[0], ls="--", ymin=0.7, color="black")
    axes[1].set_yscale("log")

        #axes[1].set_title("$\dot{m} = %d$, $i = %.1f$ $\Delta i$ = %.1f " % (mdot, incl, dincl))
#axes[1].set_ylim(bottom=1e-4)
axes[1].legend()
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("$L$ (Eddington)")
outfile = "%s/i%d_dincl%d_m%d.png" % (outdir, incl, dincl, mdot)
plt.savefig(outfile, facecolor="white", bbox_inches="tight")
print("Outputs saved to %s" %outfile)
