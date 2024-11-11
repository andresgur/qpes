# Script to fit the lightcurves (in Edd units). You need to point the script to the location of the GRHMD outputs
from astropy.io import fits
import numpy as np
import emcee
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os,warnings, time
from math import sin, cos, exp
import corner
from datetime import datetime
from shutil import copyfile

home = os.getenv("HOME")

if os.path.isfile(f"{home}/.config/matplotlib/stylelib/paper.mplstyle"):
    plt.style.use(f"{home}/.config/matplotlib/stylelib/paper.mplstyle")

mdots = np.array([2, 7, 12, 24])

PATH_TO_SIMUS = f"{home}/Documents/papers/qpes/simulations_output"

FILE = f"{PATH_TO_SIMUS}/Lx_Mdot"

mdot_files = [np.genfromtxt(FILE + "%d.txt" % mdot, names=["i_rad", "flux"]) for mdot in mdots]

#mdot = mdots[0]
#flux_i = mdot_files[0]

#print("Using mdot = %d" % mdot)

# read the flux vs i file
#flux_i = np.genfromtxt(FILE, names=["i_rad", "flux"])

TO_RAD = np.pi / 180
two_pi = 2 * np.pi

def get_mdot_file(input_mdot):
    """Casts input mdot into one of the available mdot values and retrieves the corresponding values"""
    return mdot_files[np.argmin(np.abs(input_mdot - mdots))]

def harm_model(params, times, rescale=True): # mdot=2, 
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
    # if your object is far from i ~0 then you can comment this out for speed
    if (incl - dincl) < 0.0 and incl!=0 and rescale:
        min_i = incl - dincl
        max_i = incl + dincl

        scaling_factor = (max_i - min_i) / (np.max(i_t) - np.min(i_t))

        #print("Scaling factor: %.2f" % scaling_factor)
        i_t = scaling_factor * (i_t - np.min(i_t)) + min_i
    #print(i_t_undercone[np.argmin(i_t_undercone)], i_t_undercone[np.argmax(i_t_undercone)])

    # if i>90, we see the undercone
    #i_t = np.where(i_t > 90, i_t - 90, i_t)
    i_t_undercone = np.abs(np.pi - np.abs(i_t))

    flux_i = get_mdot_file(mdot)

    #flux_i = np.genfromtxt(FILE + "%d.txt" % casted_mdot, names=["i_rad", "flux"])
    #i_t_undercone = np.where(i_t_undercone > 90, i_t_undercone - 90, i_t_undercone)
    # find the fluxes at the calculate inclinations
    #i_file = np.searchsorted(flux_i["i_rad"], i_t) - 1
    #model_rates =  flux_i["flux"][i_file]#np.interp(np.abs(i_t), flux_i["i_rad"], flux_i["flux"])
    model_rates = np.interp(np.abs(i_t), flux_i["i_rad"], flux_i["flux"])
    #i_angles = [np.argmin(np.abs(i - flux_i["i_rad"])) for i in np.abs(i_t)]
    #model_rates = flux_i["flux"][i_angles]
    #i_file_undercone = np.searchsorted(flux_i["i_rad"], i_t_undercone) - 1
    #undercone =  flux_i["flux"][i_file_undercone]
    undercone = np.interp(np.abs(i_t_undercone), flux_i["i_rad"], flux_i["flux"])
    #i_angles = [np.argmin(np.abs(i - flux_i["i_rad"])) for i in np.abs(i_t_undercone)]
    #undercone = flux_i["flux"][i_angles]
    return model_rates + A + undercone


def chi_square(params, data, errors, times):
    model = harm_model(params, times)
    return np.sum( ( (data-model) / errors)**2)


def neg_chi_square(params, data, errors, times, parambounds):

    if not np.all(np.logical_and(parambounds[:, 0] <= params, params <= parambounds[:, 1])):
        return -np.inf
    # if the inclination + di is above 180 it flips around
    elif params[2] + params[3] > 180:
        return -np.inf
    return -chi_square(params, data, errors, times)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Fit a lightcurve using the output of HARM simulations and the precession model of SS433')
    ap.add_argument("input_lc", nargs=1, help="Input lightcurve in fits format", type=str)
    ap.add_argument("-c", "--cores", nargs="?", default=12, help="Number of cores", type=int)
    ap.add_argument("-o", "--outdir", nargs="?", default="", help="Output dir", type=str)
    ap.add_argument("-p", "--period", nargs="?", default=None, help="Start period guess in seconds", type=float)
    ap.add_argument("--converge", action="store_false", help="Whether to stop the chains when converge is reached")
    ap.add_argument("--max_n", nargs="?", default=50000, help="Maximum number of samples for the chains", type=int)
    ap.add_argument("--tmin", nargs="?", default=0, help="Minimum time to truncate the lightcurve", type=float)
    ap.add_argument("--tmax", nargs="?", default=np.inf, help="Maximum time to truncate the lightcurve", type=float)
    args = ap.parse_args()

    MAX_N = args.max_n

    input_lightcurve = args.input_lc[0]

    lightcurve = fits.open(input_lightcurve)

    datetimestr = datetime.today().strftime('%m_%d_%H_%M_%S')

    outdir = os.path.basename(input_lightcurve.replace(".lc", "")) + "_" + datetimestr

    outdir += "_%s" % args.outdir

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # copy mdot files
    [copyfile(FILE + "%d.txt" % mdot, f"{outdir}/" + os.path.basename(FILE + "%d.txt" % mdot)) for mdot in mdots]

    err = lightcurve[1].data["ERROR"]

    rate = lightcurve[1].data["RATE"]

    nans = np.isnan(err)
    good_points = (~nans) & (err!=0) & (rate > 0)

    times = lightcurve[1].data["TIME"]
    if args.tmin is not None:
        good_points = good_points & (times > args.tmin)
    if args.tmax is not None:
        good_points = good_points & (times < args.tmax)

    times = lightcurve[1].data["TIME"][good_points]

    times = times - times[0]

    rate = lightcurve[1].data["RATE"][good_points]

    datapoints = len(rate)

    print("Number of datapoints: %d" % datapoints)

    err = err[good_points]

    duration = times[-1] - times[0]

    dt = lightcurve[1].header["TIMEDEL"]

    if args.period:
        starting_period = args.period
        period_bounds = (starting_period * 0.95, starting_period * 1.05)
    else:
        starting_period = duration / 6
        period_bounds = (duration / 50, duration / 6)

    phase_bounds = (0.00001, 0.99)

    incl_bounds = (1, 89.9) # (47.5, 49) ## (48, 54)

    dincl_bounds = (0.1, 75.)#(5, 6)#

    mdot_bounds = [1, 26]

    a_bounds = (0.00001, 2)# in Edd units np.mean(rate) + np.std(rate)*2)]

    initial_params = [0.3, 70., 50., 2, np.median(rate)]# np.mean(rate)] 0.5, 70, 35,  10 ]#np.mean(rate)]


    bounds = [phase_bounds, incl_bounds, dincl_bounds, mdot_bounds, a_bounds]

    par_names = [r"$\phi$", r"$i_\mathrm{0}$", r"$\Delta i$", r"$\dot{m}$", "$A$"]

    initial_params = [starting_period] + initial_params
    par_names = ["$P$"] + par_names
    parambounds = np.array([period_bounds]  + bounds)

    print("Initial parameters:\n---------------")
    print(initial_params)

    solution = minimize(chi_square, initial_params, method="L-BFGS-B",
                        bounds=parambounds, args=(rate, err, times))
    print(solution)
    print("Solved in %d iterations" % solution.nit)
    np.savetxt("%s/parameter_fit.dat" % outdir, np.array([np.append(solution.x, solution.fun)]),
                header="\t".join(par_names) + "\tloglikehood", fmt="%.3f")
    
    model_samples = datapoints * 200 if datapoints > 1000 else 6000

    model_time = np.linspace(times[0], times[-1], model_samples)

    best_model = harm_model(solution.x, model_time)# * np.mean(rate)

    print("Best-fit params:")
    print(solution.x)

    plt.figure(figsize=(12.6, 9.8))
    plt.errorbar(times / 1000, rate,
                 yerr=err, color="black", fmt=".")
    plt.plot(model_time / 1000, best_model, color="C1")
    plt.xlabel("Time (ks)")
    plt.ylabel("L (Eddington)") # (ct/s)
    plt.savefig("%s/model_fit.png" %outdir, bbox_inches="tight",
                facecolor="white")
    plt.close()

    cores = args.cores

    nwalkers = 2 * cores

    ndim = len(par_names)

    initial_samples = np.empty((nwalkers, ndim))
    initial_params = initial_params# or solution.x

    if True:
        for i in range(nwalkers):
            accepted = False

            while not accepted:
                # Generate random values centered around the best-fit parameters
                perturbed_params = np.random.normal(initial_params, np.abs(initial_params) * 0.3)

                # Check if the perturbed parameters are within the bounds
                if np.all(np.logical_and(parambounds[:, 0] <= perturbed_params, perturbed_params <= parambounds[:, 1])):
                    initial_samples[i] = perturbed_params
                    accepted = True

        autocorr = []
        every_samples = 1000
        old_tau = np.inf
        print("Initial chain samples")
        print(initial_samples.T)
        start = time.time()
        with Pool(cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, neg_chi_square,
                                            pool=pool, args=(rate, err, times, parambounds),)
            if args.converge:
                for sample in sampler.sample(initial_samples, iterations=MAX_N, progress=False):
                    # Only check convergence every 100 steps
                    if sampler.iteration % every_samples:
                        continue

                    # Compute the autocorrelation time so far
                    # Using tol=0 means that we'll always get an estimate even
                    # if it isn't trustworthy
                    tau = sampler.get_autocorr_time(tol=0)
                    autocorr.append(np.mean(tau))

                    # Check convergence
                    converged = np.all(tau * 100 < sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                    if converged:
                        print("Convergence reached after %d samples!" % sampler.iteration)
                        break
                    old_tau = tau
            else:
                converged = False
                sampler.run_mcmc(initial_samples, MAX_N, progress=True)
                tau = sampler.get_autocorr_time(tol=0)

        end = time.time()
        time_taken = (end - start ) / 60
        acceptance_ratio = sampler.acceptance_fraction
        print("Acceptance ratio: (%)")
        print(acceptance_ratio)
        print("Correlation parameters:")
        print(tau)
        mean_tau = np.mean(tau)
        print("Mean correlation time:")
        print(mean_tau)

        if not converged:
            warnings.warn("The chains did not converge!")
            # default values
            thin = 1
            discard = 0

        else:
            discard = int(mean_tau * 5)
            if discard > MAX_N:
                discard = int(mean_tau * 2.5)
            thin = int(mean_tau / 2)
            print(f"Chains converged after {sampler.iteration}")

        fig = plt.figure()
        n = np.arange(1, len(autocorr) + 1)
        plt.plot(n, autocorr, "-o")
        plt.ylabel("Mean $\\tau$")
        plt.xlabel("Number of steps")
        plt.savefig("%s/autocorr.png" %outdir, dpi=100)
        plt.close(fig)

        chain = sampler.get_chain(flat=True)
        median_values = np.median(chain, axis=0)

        chain_fig, axes = plt.subplots(ndim, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0})
        if len(np.atleast_1d(axes))==1:
            axes = [axes]
        for param, parname, ax, median in zip(chain.T, par_names, axes, median_values):
            ax.plot(param, linestyle="None", marker="+", color="black")
            ax.set_ylabel(parname.replace("kernel:", "").replace("log_", ""))
            ax.axhline(y=median)
            ax.axvline(discard * nwalkers, ls="--", color="red")

        axes[-1].set_xlabel("Step Number")
        chain_fig.savefig("%s/chain_samples.png" % outdir, bbox_inches="tight",
                          dpi=100)
        plt.close(chain_fig)
        print("Discarding the first %d samples" % discard)
        # calculate R stat
        samples = sampler.get_chain(discard=discard)

        whithin_chain_variances = np.var(samples, axis=0) # this has nwalkers, ndim (one per chain and param)

        samples = sampler.get_chain(flat=True, discard=discard)
        between_chain_variances = np.var(samples, axis=0)
        print("R-stat (values close to 1 indicate convergence)")
        print(whithin_chain_variances / between_chain_variances[np.newaxis, :]) # https://stackoverflow.com/questions/7140738/numpy-divide-along-axis
        
        samples = None # free memory

        final_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        loglikes = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
        # save samples
        print("Storing samples...")
        outputs = np.vstack((final_samples.T, loglikes))

        header_samples = "\t".join(par_names) + "\tloglikehood"

        np.savetxt("%s/samples.dat" % outdir, outputs.T, delimiter="\t", fmt="%.5f",
                  header=header_samples)
        header = ""
        outstring = ''

        best_loglikehood = np.argmax(loglikes)

        for i, parname in enumerate(par_names):
            q_16, q_50, q_84 = corner.quantile(final_samples[:,i], [0.16, 0.5, 0.84]) # your x is q_50
            header += "%s\t" % parname
            dx_down, dx_up = q_50-q_16, q_84-q_50
            decimals = int(np.abs(np.floor(np.log10(np.abs(dx_down))))) + 1
            outstring += '%.*f$_{-%.*f}^{+%.*f}$\t' % (decimals, q_50, decimals, dx_down, decimals, dx_up)
            plt.figure()
            par_vals = final_samples[:,i]
            plt.scatter(par_vals, loglikes)
            plt.scatter(par_vals[best_loglikehood], loglikes[best_loglikehood],
                        label="%.2f, L = %.2f" % (par_vals[best_loglikehood], loglikes[best_loglikehood]))
            plt.legend()
            plt.xlabel("%s" % parname)
            plt.ylabel("$L$")
            plt.savefig("%s/%s.png" % (outdir, parname), dpi=100)
            plt.close()

        header += "loglikehood\tdof"
        median_parameters = np.median(final_samples, axis=0)
        distances = np.linalg.norm(final_samples - median_parameters, axis=1)
        closest_index = np.argmin(distances)
        median_log_likelihood = loglikes[closest_index]
        dof = len(times) - ndim
        outstring += "%.3f\t%d" % (median_log_likelihood, dof)
        out_file = open("%s/parameter_medians.dat" % (outdir), "w+")
        out_file.write("%s\n%s" % (header, outstring))
        out_file.close()

        ranges = np.ones(ndim) * 0.90
        print("Storing corner plot...")
        corner_fig = corner.corner(final_samples, labels=par_names, title_fmt='.2f', #range=ranges,
                                   quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                   title_kwargs={"fontsize": 20}, max_n_ticks=3, labelpad=0.08,
                                   levels=(1 - np.exp(-0.5), 1 - exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
        corner_fig.savefig("%s/corner_fig.png" % outdir)
        plt.close(corner_fig)

        print("Plotting samples...")
        n_samples = 1500
        models = np.ones((n_samples, len(model_time)))

        fig, ax = plt.subplots(figsize=(12.6, 9.8))
        ax.errorbar(times, rate, yerr=err, color="black",
                    ls="solid", fmt=".", ecolor="gray", capsize=2.5)
        for index, sample in enumerate(final_samples[np.random.randint(len(final_samples), size=n_samples)]):
            models[index] = harm_model(sample, model_time)
            ax.plot(model_time, models[index], alpha=0.5, color="C1")
        plt.xlabel("Time (s)")
        plt.ylabel("L (Edd)")
        plt.savefig("%s/mcmc_samples.png" % outdir, bbox_inches="tight")
        plt.close(fig)

        m = np.nanpercentile(models, [16, 50, 84], axis=0)

        fig, (model_ax, res_ax) = plt.subplots(2, 1, figsize=(12.6, 12.6),
                                        sharex=True, gridspec_kw={"hspace":0, "height_ratios":[3, 1]})
        model_ax.errorbar(times / 1000, rate, yerr=err, color="black",
                          ls="solid", fmt=".", ecolor="gray", capsize=2.5)
        # model
        model_ax.plot(model_time / 1000, m[1], color="C1", lw=2, ls="solid")
        model_ax.fill_between(model_time[:-1] / 1000, m[0][:-1], m[2][:-1], alpha=0.4, color="C1", zorder=-10)
        model_ax.set_ylabel("L (Edd)")

        params_max_loglikehood = final_samples[best_loglikehood]


        best_model = harm_model(params_max_loglikehood, times)
        res = (rate - best_model)
        res_ax.errorbar(times/ 1000, res, yerr=err, color="black", ecolor="gray", capsize=2.5)
        res_ax.axhline(0, color="C1", ls="--")
        plt.xlabel("Time (ks)")
        plt.ylabel("Data - Model", labelpad=-3)
        fig.savefig("%s/mcmc_mean.png" % (outdir), bbox_inches="tight")
        plt.close(fig)
        max_params_string = ["%.*f" % (decimals, param) for param in params_max_loglikehood]
        outstring = "\t".join(max_params_string)
        outstring += "\t%.3f\t%d" % (loglikes[best_loglikehood], dof)
        out_file = open("%s/parameter_max.dat" % (outdir), "w+")
        out_file.write("%s\n%s" % (header, outstring))
        out_file.close()

    print("Took %.3f minutes for %d samples on %d cores" % (time_taken, sampler.iteration, cores))
    print("Outputs stored to %s" % outdir)
