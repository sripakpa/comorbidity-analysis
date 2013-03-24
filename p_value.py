#!/usr/bin/env python
"""
SYNOPSIS
    This script computes the p-value

DESCRIPTION
    This script computes the p-value.
    TODO: add details!

EXAMPLES

    # Use all default parameters
    python p_value.py

    # Select disease pairs.
    python p_value.py --d1 "Breast cancer (female)" --d2 "Epilepsy"

    python p_value.py --d1 "Attention deficit" --d2 "Epilepsy"

    # Select tau1 and tau2 values 
    python p_value.py --tau1 6 --tau2 6

    # Select pravalence normalization method
    python p_value.py --norm_prval_method avg

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
from scipy.stats import chi2

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


    if options.disease1 != None and options.disease2 != None:
        D1_list = [options.disease1]
        D2_list = [options.disease2]
    else:
        diseases = database.get_diseases()
        diseases.sort()
        filtered_diseases = __filter_diseases(diseases, database)
        D1_list = filtered_diseases
        D2_list = filtered_diseases

    f = open("p_value_%s.txt" % options.norm_prval_method, 'w')

    f.write("disease1, disease2, ")
    f.write("coop_log_likelihood, indep_log_likelihood, ")
    f.write("log_likelihood_ratio, p_value\n")


    for D1 in D1_list:
        for D2 in D2_list:

            if options.disease1 == None or options.disease2 == None:
                if D1 >= D2: continue

            print
            print "--------------------------------------------------"
            print "D1 = %s, D2 = %s" % (D1, D2)


            # Query patients data
            D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True)
            noD1D2_patients = database.query_emr_data(["not " + D1, "not " + D2])


            raw_count = np.zeros(4, dtype=np.float)

            raw_count[0] = len(database.query_emr_data(["not " + D1, "not " + D2]))
            raw_count[1] = len(database.query_emr_data([D1, "not " + D2]))
            raw_count[2] = len(database.query_emr_data([D2, "not " + D1]))
            raw_count[3] = len(database.query_emr_data([D2, D1]))

            # Create a function that will normalize the raw EMR data to match the
            # general population disease prevalence.
            prval_norm_func = create_prval_norm_func(D1, D2,
                                                     database.get_disease2count(),
                                                     database.get_tot_patient_count(),
                                                     options.prevalence_file,
                                                     options.norm_prval_method,
                                                     options.verbose)

            norm_count = prval_norm_func(raw_count)

            print "raw_count:"
            print "D0: %7.1f, " % raw_count[0],
            print "D1: %7.1f, " % raw_count[1],
            print "D2: %7.1f, " % raw_count[2],
            print "D12: %7.1f, " % raw_count[3],
            print "E[D12]: %7.1f" % (raw_count[1] *
                                     raw_count[2] /
                                     np.sum(raw_count))
            print "norm_count:"
            print "D0: %7.1f, " % norm_count[0],
            print "D1: %7.1f, " % norm_count[1],
            print "D2: %7.1f, " % norm_count[2],
            print "D12: %7.1f, " % norm_count[3],
            print "E[D12]: %7.1f" % (norm_count[1] *
                                     norm_count[2] /
                                     np.sum(norm_count))

            sys.stdout.flush()

            # Conditional probabilities: P(phi(t) | phi(infty))
            joint_age_of_onset = JointAgeOfOnset(D1orD2_patients)

            joint_age_of_onset_funcs = joint_age_of_onset.get_funcs()

            # Patient counts of as a distribution of the final age value.
            joint_final_age = JointFinalAge(D1orD2_patients, noD1D2_patients)

            final_age_array = joint_final_age.get_final_age_array()
            patient_counts = joint_final_age.get_patient_counts()


            # Independent model
            inde_optimized_param = (
                __get_optimized_parameters_wrapper(options.tau1,
                                                   options.tau2,
                                                   "independent",
                                                   options.threshold_type,
                                                   options.opt_method,
                                                   options.verbose,
                                                   joint_age_of_onset_funcs,
                                                   patient_counts,
                                                   final_age_array,
                                                   prval_norm_func,
                                                   rng))

            inde_log_likelihood = inde_optimized_param["log_likelihood"]

            min_log_likelihood = inde_log_likelihood - 1.0

            # Cooperative model
            coop_optimized_param = (
                __get_optimized_parameters_wrapper(options.tau1,
                                                   options.tau2,
                                                   "cooperation",
                                                   options.threshold_type,
                                                   options.opt_method,
                                                   options.verbose,
                                                   joint_age_of_onset_funcs,
                                                   patient_counts,
                                                   final_age_array,
                                                   prval_norm_func,
                                                   rng,
                                                   min_log_likelihood))

            coop_log_likelihood = coop_optimized_param["log_likelihood"]

            log_likelihood_ratio = 2.0 * (coop_log_likelihood -
                                          inde_log_likelihood)

            print "coop_log_likelihood= ", coop_log_likelihood
            print "inde_log_likelihood= ", inde_log_likelihood
            print "log_likelihood_ratio=", log_likelihood_ratio

            dof = 1  # degree of freedoms
            p_value = 1.0 - chi2.cdf(log_likelihood_ratio, dof)

            text = "%s, %s, " %(D1, D2)
            text += "%.3f, %.3f, " %(coop_log_likelihood, inde_log_likelihood)
            text += "%.5f, %s" % (log_likelihood_ratio, p_value)

            print text
            f.write(text + '\n')

    f.close()

def __filter_diseases(diseases, database):
    """Filter for diseases where patients counts for disease is non-zero."""

    filtered_Ds = []

    for D in diseases:

        D_patients = database.query_emr_data(D)
        if len(D_patients) != 0: filtered_Ds.append(D)

    return filtered_Ds

def __get_optimized_parameters_wrapper(tau1, 
                                       tau2,
                                       overlap_type,
                                       threshold_type,
                                       opt_method,
                                       verbose,
                                       joint_age_of_onset_funcs,
                                       patient_counts,
                                       final_age_array,
                                       prval_norm_func,
                                       rng,
                                       min_log_likelihood=None):
    """TODO: add description"""

    # Setup the genetic_penetrance model to compute the age-integrated
    # phenotype probabilities: P(phi(infty) ; rho1, rho2, rho12)
    genetic_penetrance = GeneticPenetrance(tau1,
                                           tau2,
                                           overlap_type,
                                           threshold_type)

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


    (rho1_max, rho2_max) = (1.0 * tau1, 1.0 * tau2)

    rho12_max = (rho1_max + rho2_max) / 2.0

    while True:

        optimized_param = __get_optimized_parameters(log_likelihood_func,
                                                     log_likelihood_fprime,
                                                     rng,
                                                     rho1_max,
                                                     rho2_max,
                                                     rho12_max,
                                                     overlap_type,
                                                     opt_method,
                                                     verbose)

        if min_log_likelihood == None: break

        if optimized_param["log_likelihood"] > min_log_likelihood: break


    return optimized_param


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

    sign = -1.0 # maximize instead of minimize.

    minimize_test = __get_test_optimization_func(opt_method)

    start_time = time.time()
    xopt, allvec = minimize_test(log_likelihood_func,
                                 [log_rho1, log_rho2, log_rho12],
                                 args=(log_rho, sign,),
                                 fprime=log_likelihood_fprime)

    print "overlap_type = %s" % overlap_type 
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
    opt_param["rho12"] = np.exp(xopt[2])
    opt_param["log_likelihood"] = log_likelihood_func(xopt)

    if options.overlap_type == "independent": opt_param["rho12"] = 0.0

    return opt_param

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
    elif opt_method == 'BFGS':
        def minimize(f, x0, args=(), fprime=None):
            return optimize.fmin_bfgs(f, x0, args=args, gtol=gtol_test,
                                      retall=True)
    else:
        raise ValueError("Unvalid opt_method %s" % opt_method)

    return minimize


if __name__ == '__main__':

    start_time = time.time()

    usage = """test_optimize_log_likelihood.py -i <emr_data_file> 
            --d1 <1st_disease> --d2 <2nd_disease>"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data", action="store", type="string",
                      default="data/test_EMR_dataset.csv",
                      dest="emr_data_file",
                      help="RAW EMR data file")

    parser.add_option("--d1", action="store", type="string",
                      default=None, dest="disease1",
                      help="name of the first disease.")

    parser.add_option("--d2", action="store", type="string",
                      default=None, dest="disease2", 
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
