# Script to convert count rates to Eddington luminosities based on the luminosities from Table A1 and A2 from the paper.
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from scipy.interpolate import interp1d


def eddington_luminosity(M):
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: float
            Mass in solar masses
        Returns the Eddington luminosity
    """
    return 1.26 * M * 10.**38

def rate_to_edd(lums, countrates, inputrate):
    func = interp1d(countrates, lums, fill_value="extrapolate")
    edd_lum = func(inputrate)
    return edd_lum


mass_gsn = 10.**5.99
# quiescence, rise1, rise2 and peak
lums = np.array([3.7019, 4.76, 11.87, 24.08]) * 10.**41 / eddington_luminosity(mass_gsn) # 3.7019, 
countrates = np.array([0.1159, 0.279, 0.9070, 1.495]) # 0.1159,
name = "GSN069"
input_lightcurve = "/home/andresgur/xray_data/QPEs/GSN069/0831790701/outputpn/lightcurves/300/PN_source_corr_broadband_sub.lc"

# ERO QPE-2
#name = "ERO-QPE2"
# decay, rise, peak
#mass_eroqpe2 = 10.**4.96
#lums = np.array([3.14, 5.26, 10.19]) * 10.**41 / eddington_luminosity(mass_eroqpe2) # 5.26, 3.959
#countrates = np.array([3.074e-02, 1.41e-01, 3.466e-01]) # 1.410e-01,  1.41e-01, 1.235e-01
#input_lightcurve = "/home/andresgur/xray_data/QPEs/ERO-QPE2/0872390101/outputpn/lightcurves/300/PN_source_corr_broadband_sub.lc"

lightcurve = fits.open(input_lightcurve)

outfile = input_lightcurve.replace(".lc", "_Edd.lc")

err = lightcurve[1].data["ERROR"]
rate = lightcurve[1].data["RATE"]

# Initialize LEDD with NaNs
LEDD = np.full_like(rate, np.nan)
LEDD_err = np.full_like(rate, np.nan)

nans = np.isnan(err)
good_points = (~nans) & (err!=0) & (rate > 0)

times = lightcurve[1].data["TIME"][good_points]

err = lightcurve[1].data["ERROR"][good_points]

rate = lightcurve[1].data["RATE"][good_points]

edd_lum = rate_to_edd(lums, countrates, rate)
edd_lum_err = (rate_to_edd(lums, countrates, rate + err) - rate_to_edd(lums, countrates, rate - err)) / 2
#edd_lum_err = edd_lum - rate_to_edd(lums, countrates, rate - err)
#edd_lum_err = (rate_to_edd(lums, countrates, rate + err) - rate_to_edd(lums, countrates, rate - err)) / 2

# Fill in the LEDD array with computed values for good points
LEDD[good_points] = edd_lum
LEDD_err[good_points] = edd_lum_err

lightcurve[1].data["RATE"] = LEDD
lightcurve[1].data["ERROR"] = LEDD_err
lightcurve.writeto(outfile, overwrite=True)
#edd_err = rate_to_edd(lums, countrates, err)
#lightcurve[1].data["RATE"] = edd_rates
#lightcurve[1].data["ERROR"] = rate_to_edd(lums, countrates, err)
#rate = lightcurve[1].data["RATE"]
#times = lightcurve[1].data["TIME"]

plt.figure(figsize=(12.6, 9.8))
plt.ylabel("L (Eddington)")
plt.errorbar(times / 1000, edd_lum, yerr=edd_lum_err, markerfacecolor="None",
             color="black", fmt="o", markersize=10)


twiny = plt.gca().twinx()
twiny.errorbar(times / 1000, rate,
                yerr=err, color="blue", fmt=".")
twiny.set_ylabel("Rate (ct/s)")
twiny.yaxis.label.set_color("blue")
plt.xlabel("Time (ks)")

plt.savefig(f"{name}_converted.png")
