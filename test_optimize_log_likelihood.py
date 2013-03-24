#!/usr/bin/env python
"""
SYNOPSIS
    This script test the optimization of the log likelihood function.

DESCRIPTION
    This script test the optimization of the log_likelihood function for
    explaining the EMR database using the genetic penetrance models proposed
    by Rzhetsky et al. 2007.

    Model parameters (rho1, rho2 and rho12) are initialized at <num_paths>
    different randomly chosen location. The model parameters are optimize
    using coordinate gradient or gradient descent algorithm to maximize the 
    log-likelihood and the optimization paths are saved.

    User set num_paths with --num_paths flag (default is 5).

    Optimization are performed with scipy.optimize package.

    User set the optimization method with --opt_method flag. Valid methods are
    'BFGS' (default), 'Nelder-Mead', 'Powell' and'CG'.

    Our testing showed that:
        (1) 'BFGS' is most robust
        (2) 'Powell' sometime give numerical errors.
        (3) 'CG' (conjugate gradient) sometime give numerical errors.

    The script generates as output a plot with the following subplots:

    (1) Root-mean-square-deviation (RMSD) of the model parameters (rho1, rho2
        and rho12) from their optimal values at each iteration along the 
        optimization paths.

        Since the log_likelihood function is convex, all the optimization paths 
        should converge to the (same) global maximum (RMSD should approach 0).

    (2) Contour plot of log-likelihood function along the rho1-rho2 plane, at
        the optimal rho12 value. Optimization paths are then overlayed on
        the contour plot.

    (3) Same of (2) but along the rho1-rho12 plane at the optimal rho2 value.

    (4) Same of (2) but along the rho2-rho12 plane at the optimal rho1 value.

    Other optional command line options:

        User can set the value of genetic penetrance model's parameters: 
            (1) --tau1 (default is 1)
            (2) --tau2 (default is 1)
            (3) --overlap_type (default is coorperative)
            (4) --threshold_type (default is sharp)

        User can set the EMR dataset and disease pair D1 and D2:
            (1) --emr_data (default is data/test_EMR_dataset.csv)
            (2) --d1 (default is "Breast cancer (female)")
            (3) --d2 (default is "Epilepsy")

EXAMPLES

    # Use all default parameters
    python test_optimize_log_likelihood.py

    # Select tau1 and tau2 values 
    python test_optimize_log_likelihood.py --tau1 6 --tau2 6

    # Select indepedent penetrance model type
    python test_optimize_log_likelihood.py --overlap_type independent

    # Use BFGS optimization method
    python test_optimize_log_likelihood.py --opt_method BFGS

    # Select pravalence normalization method
    python test_optimize_log_likelihood.py --norm_prval_method sum

    # Select disease pairs.
    python test_optimize_log_likelihood.py \
        --d1 "Breast cancer (female)" --d2 "Epilepsy"

    python test_optimize_log_likelihood.py \
        --d1 "Attention deficit" --d2 "Autism"

    # Select EMR database
    python test_optimize_log_likelihood.py \
        --emr_data data/test_EMR_dataset.csv

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from emr_database_class import EMRDatabase
from genetic_penetrance_class import GeneticPenetrance
from joint_age_of_onset_class import JointAgeOfOnset
from joint_final_age_class import JointFinalAge
from norm_prevalence_func import create_prval_norm_func
from log_likelihood_func import create_log_likelihood_func
from log_likelihood_func import create_log_likelihood_fprime

def main():

    global options

    # Pseudo-random number generator.
    rng = np.random.RandomState(50)  # fixed seed
    if options.use_random_seed: self.rng.seed()  

    # Import EMR data into database
    database = EMRDatabase()

    database.import_data(options.emr_data_file,
                         options.diseases_file,
                         options.code2disease_file)

    D1 = options.disease1
    D2 = options.disease2

    # Query patients data
    D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True)
    noD1D2_patients = database.query_emr_data(["not " + D1, "not " + D2])

    # Setup the genetic_penetrance model to compute the age-integrated
    # phenotype probabilities: P(phi(infty) ; rho1, rho2, rho12)
    genetic_penetrance = GeneticPenetrance(options.tau1,
                                           options.tau2,
                                           options.overlap_type,
                                           options.threshold_type)

    # Create a function that will normalize the raw EMR data to match the
    # general population disease prevalence.
    prval_norm_func = create_prval_norm_func(D1, D2,
                                             database.get_disease2count(),
                                             database.get_tot_patient_count(),
                                             options.prevalence_file,
                                             options.norm_prval_method,
                                             options.verbose)

    # Conditional probabilities: P(phi(t) | phi(infty))
    joint_age_of_onset = JointAgeOfOnset(D1orD2_patients)

    joint_age_of_onset_funcs = joint_age_of_onset.get_funcs()

    # Patient counts of as a distribution of the final age value.
    joint_final_age = JointFinalAge(D1orD2_patients, noD1D2_patients)

    final_age_array = joint_final_age.get_final_age_array()
    patient_counts = joint_final_age.get_patient_counts()

    # Function to evaluate log-likelihood at given rho1, rho2 and rho12.
    log_likelihood_func = create_log_likelihood_func(genetic_penetrance,
                                                     joint_age_of_onset_funcs,
                                                     patient_counts,
                                                     final_age_array,
                                                     prval_norm_func)

    # Function to evaluate the derivative of the log-likelihod wrt
    # to rho1, rho2 and rho12.
    log_likelihood_fprime = (
        create_log_likelihood_fprime(genetic_penetrance,
                                     joint_age_of_onset_funcs,
                                     patient_counts,
                                     final_age_array,
                                     prval_norm_func))


    (rho1_max, rho2_max) = (3.0 * options.tau1, 3.0 * options.tau2)

    rho12_max = (rho1_max + rho2_max) / 2.0

    optimized_param = __get_optimized_parameters(log_likelihood_func,
                                                 log_likelihood_fprime,
                                                 rng,
                                                 rho1_max,
                                                 rho2_max,
                                                 rho12_max,
                                                 options.overlap_type,
                                                 options.opt_method,
                                                 options.verbose)

    optimization_paths = __compute_optimization_paths(log_likelihood_func,
                                                      log_likelihood_fprime,
                                                      rng,
                                                      rho1_max, 
                                                      rho2_max,
                                                      rho12_max,
                                                      options.overlap_type,
                                                      options.opt_method,
                                                      options.num_paths,
                                                      options.verbose)

    plot = __plot_contour(log_likelihood_func,
                          optimized_param,
                          optimization_paths,
                          genetic_penetrance,
                          options.norm_prval_method,
                          D1, D2,
                          options.verbose)

    plt.show()

def __get_optimized_parameters(log_likelihood_func, log_likelihood_fprime,
                               rng, rho1_max, rho2_max, rho12_max,
                               overlap_type, opt_method, verbose):
    """Compute parameter that maximize the log_likelihood value.

    Returns
    -------
    opt_param: a dictionary
        Dictionary containing the parameter that maximize the log_likelihood.
        The key-value pairs are :
            "rho1" : value of rho1 (float)
            "rho2" : value of rho2 (float)
            "rho12" : value of rho12 (float)
    """

    # uniform distribution over (0, 1]
    (rho1, rho2, rho12) = rng.rand(3) + 1.0e-20

    (rho1, rho2, rho12) = (rho1 * rho1_max,
                           rho2 * rho2_max,
                           rho12 * rho12_max)

    print "start: rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (rho1,
                                                             rho2,
                                                             rho12)

    log_rho = True
    log_rho1 = math.log(rho1)
    log_rho2 = math.log(rho2)
    log_rho12 = math.log(rho12)

    # if overlap_type == "independent": rho12 = 0.0

    sign = -1.0 # maximize instead of minimize.

    minimize_test = __get_test_optimization_func(opt_method)

    start_time = time.time()
    xopt, allvec = minimize_test(log_likelihood_func,
                                 [log_rho1, log_rho2, log_rho12],
                                 args=(log_rho, sign,),
                                 fprime=log_likelihood_fprime)

    print " " * 8, 'minimize_time = %.3f secs' % (time.time() - start_time)
    print " " * 8, "rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (np.exp(xopt[0]),
                                                               np.exp(xopt[1]),
                                                               np.exp(xopt[2]))
    if verbose:
        for n, point in enumerate(allvec):
            print " " * 8,
            print "%3d: rho1, rho2, rho12, log_likelihood = " % n,
            print "%.3f " % np.exp(point[0]),
            print "%.3f " % np.exp(point[1]),
            print "%.3f " % np.exp(point[2]),
            print "%.3f" % log_likelihood_func(point)
        print

    opt_param = dict()

    opt_param["rho1"] = np.exp(xopt[0])
    opt_param["rho2"] = np.exp(xopt[1])

    if options.overlap_type == "independent": 
        opt_param["rho12"] = 0.0
    else:
        opt_param["rho12"] = np.exp(xopt[2])


    return opt_param

def __compute_optimization_paths(log_likelihood_func, log_likelihood_fprime,
                                 rng, rho1_max, rho2_max, rho12_max, 
                                 overlap_type, opt_method, num_paths, verbose):
    """Compute optimization paths starting from random points in the parameter
    space.

    Returns
    -------
    paths : list of dictionary
        Each paths[i] is a dictionary contain information about the ith
        optimization path.

        path : dictionary
            Information at all iterations along the optimization path stored
            in the following key-value pairs:
                "rho1" : list of rho1 values.
                "rho2" : list of rho2 values. 
                "rho12" : list of rho12 values. 
                "log_likelihood" : list of log_likelihood value.
            The jth element of each list store the the value at the jth 
            optimization iteration step.
    """

    paths = []

    for n in range(num_paths):

        # uniform distribution over (0, 1]
        (rho1, rho2, rho12) = rng.rand(3) + 1.0e-20

        (rho1, rho2, rho12) = (rho1 * rho1_max,
                               rho2 * rho2_max,
                               rho12 * rho12_max)

        print "start: rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (rho1,
                                                                 rho2,
                                                                 rho12)

        log_rho = True
        log_rho1 = math.log(rho1)
        log_rho2 = math.log(rho2)
        log_rho12 = math.log(rho12)

        # if overlap_type == "independent": rho12 = 0.0

        sign = -1.0 # maximize instead of minimize.

        minimize_test = __get_test_optimization_func(opt_method)

        start_time = time.time()
        xopt, allvec = minimize_test(log_likelihood_func,
                                     [log_rho1, log_rho2, log_rho12],
                                     args=(log_rho, sign,),
                                     fprime=log_likelihood_fprime)
        print " " * 8,
        print 'minimize_time = %.3f secs' % (time.time() - start_time)
        print " " * 8,
        print "rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (np.exp(xopt[0]),
                                                          np.exp(xopt[1]),
                                                          np.exp(xopt[2]))
        if verbose:
            for n, point in enumerate(allvec):
                print " " * 8,
                print "%3d: rho1, rho2, rho12, log_likelihood = " % n,
                print "%.3f " % np.exp(point[0]),
                print "%.3f " % np.exp(point[1]),
                print "%.3f " % np.exp(point[2]),
                print "%.3f" % log_likelihood_func(point)
            print

        path = {'rho1' : [], 'rho2': [], 'rho12': [], 'log_likelihood': []}
        for n, point in enumerate(allvec):
            path['rho1'].append(np.exp(point[0]))
            path['rho2'].append(np.exp(point[1]))

            if options.overlap_type == "independent":
                path['rho12'].append(0)
            else:
                path['rho12'].append(np.exp(point[2]))

            path['log_likelihood'].append(log_likelihood_func(point))

        paths.append(path)

    return paths

def __plot_contour(log_likelihood_func,
                   opt_param,
                   optimization_paths,
                   genetic_penetrance,
                   norm_prval_method,
                   D1, D2,
                   verbose):
    """Plot and save the log_likelihood contour with pyplot."""

    tau1 = genetic_penetrance.get_tau1()
    tau2 = genetic_penetrance.get_tau2()
    overlap_type = genetic_penetrance.get_overlap_type()
    threshold_type = genetic_penetrance.get_threshold_type()

    # Set log_likelihood_func, optimization_paths, tau1, tau2, opt_param
    def subplot_contour(subplot, x_dim, y_dim, const_dim):

        return __subplot_contour(subplot, x_dim, y_dim,
                                 const_dim, opt_param[const_dim],
                                 log_likelihood_func,
                                 optimization_paths,
                                 tau1, tau2)

    # Define plot size in inches (width, height) & resolution(DPI)
    fig = plt.figure(figsize=(10.0, 8.0), dpi=100)

    # Axes coordinates are 0,0 is bottom left and 1,1 is upper right
    axes = fig.add_axes([0, 0, 1, 1])

    plt.clf()

    # (1) Plot param rmsd along opimization paths
    subplot = plt.subplot(2, 2, 1)

    rho_rmsd_along_paths = __compute_rho_rmsd(opt_param, optimization_paths)
    __subplot_rho_rmsd(subplot, rho_rmsd_along_paths)

    # (2) Plot contour along rho1-rho2 plane
    subplot = plt.subplot(2, 2, 2)
    (x_dim, y_dim, const_dim)= ("rho1", "rho2", "rho12")
    subplot_contour(subplot, x_dim, y_dim, const_dim)

    # (3) Plot contour along rho1-rho12 plane
    subplot = plt.subplot(2, 2, 3)
    (x_dim, y_dim, const_dim)= ("rho1", "rho12", "rho2")
    subplot_contour(subplot, x_dim, y_dim, const_dim)

    # (4) Plot contour along rho2-rho12 plane
    subplot = plt.subplot(2, 2, 4)
    (x_dim, y_dim, const_dim)= ("rho2", "rho12", "rho1")
    subplot_contour(subplot, x_dim, y_dim, const_dim)

    # Global title.
    plt.text(0.50, 0.98,
             'Test optimize log_likelihood, ' + 
             r'$\tau_1$' + ' = %d, ' % tau1 +
             r'$\tau_2$' + ' = %d, ' % tau2 +
             'type = (%s, %s)' % (overlap_type, threshold_type),
             horizontalalignment='center', verticalalignment='top',
             fontsize=15, transform = axes.transAxes)

    # Save the plot as a png image.
    fig_name = "test_optimize_log_likelihood_"
    fig_name += "%s_" % overlap_type
    fig_name += "%s_" % norm_prval_method
    fig_name += "tau1=%d_" % tau1
    fig_name += "tau2=%d" % tau2
    fig_name +=".png" 
    fig_name = fig_name.replace(" ", "_")

    if verbose: print "saving %s to file" % fig_name

    plt.savefig(fig_name, format='png') 

    return plt


def __get_test_optimization_func(opt_method):
    """Select and return the specified test optimization function.

    Use this function to experiment with different values of gtol and ftol
        gtol: float (scipy defualt is 1e-05)
            Gradient tolerance, stop when norm of gradient is less than gtol.
        ftol : float (scipy defualt is 0.0001)
            Function tolerance, relative error in func(xopt) acceptable for 
            convergence.
    """

    gtol_test = 0.001
    ftol_test = 0.001

    if opt_method == 'Nelder-Mead':
        def minimize(f, x0, args=(), fprime=None):
            return optimize.fmin(f, x0, args=args, ftol=ftol_test,
                                 retall=True)
    elif opt_method == 'Powell':
        def minimize(f, x0, args=(), fprime=None):
            return optimize.fmin_powell(f, x0, args=args, ftol=ftol_test, 
                                        retall=True)
    elif opt_method == 'CG':
        def minimize(f, x0, args=(), fprime=None):
            return optimize.fmin_cg(f, x0, args=args, gtol=gtol_test,
                                    retall=True)
        # fprime=fprime,
    elif opt_method == 'BFGS':
        def minimize(f, x0, args=(), fprime=None):
            return optimize.fmin_bfgs(f, x0, args=args, gtol=gtol_test,
                                      retall=True)
        # fprime=fprime,
    else:
        raise ValueError("Unvalid opt_method %s" % opt_method)

    return minimize


def __compute_rho_rmsd(opt_param, optimization_paths):
    """Compute and return the rmsd of rho parameters from the optimized rho
     parameters along the optimization_paths."""

    rho_rmsd_along_paths = []

    for path in optimization_paths:

        rho_rmsd_along_path = []

        for n in xrange(len(path["rho1"])):

            rho_rmsd = ((opt_param["rho1"] - path["rho1"][n]) ** 2 +
                          (opt_param["rho2"] - path["rho2"][n]) ** 2 +
                          (opt_param["rho12"] - path["rho12"][n]) ** 2)

            rho_rmsd = rho_rmsd / 3.0

            rho_rmsd = math.sqrt(rho_rmsd)

            rho_rmsd_along_path.append(rho_rmsd) 

        rho_rmsd_along_paths.append(rho_rmsd_along_path)

    return rho_rmsd_along_paths

def __subplot_rho_rmsd(subplot, rho_rmsd_along_paths):
    """Plot the rmsd of rho parameters from the optimized rho parameters along 
    the optimization_paths."""

    for rho_rmsd_along_path in rho_rmsd_along_paths:

        iterations = range(len(rho_rmsd_along_path))

        plt.plot(iterations, rho_rmsd_along_path, linestyle='-', marker= 'o')

    plt.xlabel("iterations")
    plt.ylabel(r'$\rho_1$' + ', ' + r'$\rho_2$' + ', ' + r'$\rho_{12}$ ' + \
               'RMSD convergence')

def __subplot_contour(subplot, x_dim, y_dim,
                      const_dim, const_rho,
                      log_likelihood_func,
                      optimization_paths,
                      tau1, tau2):
    """Plot log_likelihood contour along the x_dim-y_dim plane.

    Parameters
    ----------
    log_likelihood_func:
        log_likelihood func evaluate the log likelihood value of
        observing the EMR data at given [rho1, rho2, rho12]

    x_dim: string ("rho1", "rho2" or "rho12")
        parameter along the x_dim 

    y_dim: string ("rho1", "rho2" or "rho12")
        parameter along the y_dim 

    const_rho: float
        Value of the rho parameter along the const_dim.

    tau1: int
        Value of tau1 from the genetic penetrance model. Use as normalizing
        factor of rho1 value.

    tau2: int
        Value of tau2 from the genetic penetrance model.Use as normalizing
        factor of rho2 value.
    """

    X, Y, PROBS = __compute_rho_2Dgrid(log_likelihood_func,
                                       x_dim, y_dim,
                                       const_dim, const_rho,
                                       tau1, tau2)

    CS = plt.contour(X, Y, PROBS)
    plt.clabel(CS, inline=1, fontsize=10)

    for path in optimization_paths:
        plt.plot(path[x_dim], path[y_dim], linestyle=':', marker= 'o')

    # Set labels
    dim2symbols = {"rho1" : r'$\rho_1$',
                   "rho2" : r'$\rho_2$',
                   "rho12" : r'$\rho_{12}$'}

    x_symbol = dim2symbols[x_dim]
    y_symbol = dim2symbols[y_dim]
    const_symbol = dim2symbols[const_dim]

    plt.xlabel(x_symbol)
    plt.ylabel(y_symbol)

    legend = plt.legend([], title = x_symbol + '-' + y_symbol + " plane, " +
                        const_symbol + '= %.3f' % const_rho, loc='best')
    legend.get_title().set_fontsize('12')

    # Set x and y axes
    tau_map = {"rho1" : tau1, "rho2" : tau2, "rho12" : max(tau1, tau2)}

    x_tau, y_tau = (tau_map[x_dim], tau_map[y_dim])
 
    x_min, x_max = (0.0 * x_tau, 3.0 * x_tau) 
    y_min, y_max = (0.0 * y_tau, 3.0 * y_tau)

    subplot.set_xticks(np.arange(x_min, x_max + 1, x_tau))
    subplot.set_yticks(np.arange(y_min, y_max + 1, y_tau))

    subplot.set_xlim([x_min, x_max])
    subplot.set_ylim([y_min, y_max])

def __compute_rho_2Dgrid(log_likelihood_func,
                         x_dim, y_dim, 
                         const_dim, const_rho,
                         tau1, tau2):

    """Compute the log_likelihod value along a rho1 x rho2 grid.

    

    Two of the parameters are enumerated along the grid. The third parameter
    is passed in as other_rho

    Parameters
    ----------
    log_likelihood_func:
        log_likelihood func evaluate the log likelihood value of
        observing the EMR data at given [rho1, rho2, rho12]

    x_dim: string ("rho1", "rho2" or "rho12")
        parameter along the x_dim 

    y_dim: string ("rho1", "rho2" or "rho12")
        parameter along the y_dim 

    const_dim: string ("rho1", "rho2" or "rho12")
        parameter along the dimension that is not enumerated.

        For example if x_dim = "rho1" and y_dim = "rho2", then other_dim
        should be "rho12".

    const_rho: float
        Value of the rho parameter along the const_dim.

    tau1: int
        Value of tau1 from the genetic penetrance model. Use as normalizing
        factor of rho1 value.

    tau2: int
        Value of tau2 from the genetic penetrance model.Use as normalizing
        factor of rho2 value.
    """

    tau_map = {"rho1" : tau1, "rho2" : tau2, "rho12" : max(tau1, tau2)}

    x_tau, y_tau = (tau_map[x_dim], tau_map[y_dim])
    x_min, x_max, x_delta = (0.0 * x_tau, 3.0 * x_tau, 0.1 * x_tau)
    y_min, y_max, y_delta = (0.0 * y_tau, 3.0 * y_tau, 0.1 * y_tau)

    # x_tau and y_tau acts as a scaling factor of x and y dimensions.
    x_array = np.arange(x_min + 0.01, x_max, x_delta) + x_delta
    y_array = np.arange(y_min + 0.01, y_max, y_delta) + y_delta

    # Notice the switch in the order of x and y here!
    # If do not switch, then will get the tranpose of want we want!
    Y, X = np.meshgrid(y_array, x_array)

    PROBS = np.zeros([x_array.size, y_array.size], dtype=np.float64)

    dim2rhoindex = {"rho1" : 0, "rho2" : 1, "rho12" : 2}

    x_to_rho = dim2rhoindex[x_dim]
    y_to_rho = dim2rhoindex[y_dim]
    const_to_rho = dim2rhoindex[const_dim]

    for x_index, x in enumerate(x_array):
        for y_index, y in enumerate(y_array):

            rho_list = [0, 0, 0]

            rho_list[x_to_rho] = x
            rho_list[y_to_rho] = y
            rho_list[const_to_rho] = const_rho

            PROBS[x_index, y_index] = log_likelihood_func(rho_list,
                                                          log_rho=False)

    return X, Y, PROBS


if __name__ == '__main__':

    start_time = time.time()

    usage = """python test_optimize_log_likelihood.py -i <emr_data_file> 
            --d1 <1st_disease> --d2 <2nd_disease>"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data", action="store", type="string",
                      default="data/test_EMR_dataset.csv",
                      dest="emr_data_file",
                      help="RAW EMR data file")

    parser.add_option("--d1", action="store", type="string",
                      default="Breast cancer (female)", dest="disease1",
                      help="name of the first disease.")

    parser.add_option("--d2", action="store", type="string",
                      default="Epilepsy", dest="disease2", 
                      help="name of the second disease")

    parser.add_option("--diseases_file", action="store",
                      default="data/disease-list.txt", type="string",
                      dest="diseases_file",
                      help="file contain list of diseases.")

    parser.add_option("--code2disease_file", action="store", type="string",
                      default="data/code-to-disease.csv",
                      dest="code2disease_file",
                      help="CSV file mapping ICD9 code to disease.")

    parser.add_option("--norm_prval_method", action="store", type="string",
                      default="Rzhetsky",
                      dest="norm_prval_method",
                      help="select protocol to normalize the EMR prevalence.")

    parser.add_option("--prevalence_file", action="store", type="string",
                      default="data/disease-prevalence.csv",
                      dest="prevalence_file",
                      help="prevalence of disease in general population.")

    parser.add_option("--tau1", action="store", type="int",
                      default=1, dest="tau1", 
                      help="tau1 parameter of the genetic penetrance model.")

    parser.add_option("--tau2", action="store", type="int",
                      default=1, dest="tau2", 
                      help="tau2 parameter of the genetic penetrance model.")

    parser.add_option("--rho12", default=1.0, action="store", type="float",
                      dest="rho12", 
                      help="rho12 parameter of the genetic penetrance model.")

    parser.add_option("--overlap_type",  action="store",  type="string",
                      default="cooperation", dest="overlap_type",
                      help="type of genetic overlap used in the model.")

    parser.add_option("--threshold_type", action="store", type="string",
                      default="sharp", dest="threshold_type", 
                      help="type of genetic penetrance used in the model.")

    parser.add_option("--num_paths", action="store", type="int",
                      default=5, dest="num_paths", 
                      help="number of optimization paths to compute.")

    parser.add_option("--use_random_seed", action="store_true", default=False,
                      dest="use_random_seed",
                      help="random seed for pseudo-random number generator")

    parser.add_option("--opt_method", type="string",  action="store",
                      default='BFGS', dest="opt_method",
                      help="name of the optimization method.")

    parser.add_option("-v", "--verbose", action="store_true", default=True,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=True,
                      dest="verbose", help="no verbose output.")

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if options.norm_prval_method == "None":
        options.norm_prval_method = None

    if options.norm_prval_method not in [None, "Rzhetsky", "max", "min", "avg",
                                         "wts_avg", "sum","independent"]:

        raise ValueError("invalid --norm_prval_method %s " %
                         options.norm_prval_method)

    if options.overlap_type not in ["cooperation",
                                    "competition",
                                    "independent"]:
        raise ValueError("invalid --overlap_type %s " % options.overlap_type)

    if options.threshold_type not in ["sharp", "soft"]:
        raise ValueError("invalid --threshold_type %s " %
                         options.threshold_type)

    if options.opt_method not in ['Nelder-Mead', 'Powell', 'CG', 'BFGS']:
        raise ValueError("invalid --opt_method %s " % options.opt_method)

    if options.verbose:
        print "-" * 50
        print time.asctime()
        print 
        print "EMR data parameters:"
        print "--------------------"
        print "disease1: %s" % options.disease1
        print "disease2: %s" % options.disease2
        print "emr_data_file: %s" % options.emr_data_file
        print "diseases_file: %s" % options.diseases_file
        print "code2disease_file: %s" % options.code2disease_file
        print 
        print "Normalize prevalence parameters:"
        print "--------------------------------"
        print "norm_prval_method: %s" % options.norm_prval_method
        print "prevalence_file: %s" % options.prevalence_file
        print 
        print "Genetic penetrance model parameters:"
        print "------------------------------------"
        print "tau1: %s" % options.tau1
        print "tau2: %s" % options.tau2
        print "overlap_type: %s" % options.overlap_type
        print "threshold_type: %s" % options.threshold_type
        print
        print "Plotting parameters:"
        print "--------------------"
        print "num_paths: %s" % options.num_paths
        print "use_random_seed: %s" % options.use_random_seed
        print "optimization_method %s" % options.opt_method
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
