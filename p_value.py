#!/usr/bin/env python
"""
SYNOPSIS
    This script computes the statistical significance (p-value) of observing
    the EMR dataset for pair of diseases according to the genetic penetrance
    models proposed in Rzhetsky et al. 2007 [1].

DESCRIPTION

    User specify the EMR data_file with -i flag and a pair of diseases with
    --d1 and --d2 flags.

    The scripts then computes and output the p-value for the specified disease
    pair to (1) stdout and (2) "p_value.txt".

    If the user doesn't specify a disease pair, then the p-value is
    computed for all pairs of diseases with non-zero patient counts
    (disease pairs generated from the 161 disease list).

    A small p-value (e.g 0.05 significance level after Bonferroni correction),
    would indicate that the chance of observing the EMR data given the null
    hypothesis that the two diseases are not comorbid is highly unlikely and
    hence the two disorders are likely be comorbid.
    
NOTES

    Optional command line options:

    User can set the value of genetic penetrance model's parameters: 
        (1) --tau1 (default is 1)
        (2) --tau2 (default is 1)
        (3) --overlap_type (default is coorperative)
        (4) --threshold_type (default is sharp)

    User can specify how the patient counts in the EMR dataset is
    normalized using the --norm_prval_method flag. The choices are:

        (1) "Rzhetsky" (default): Use the exact normalization procedure
            proposed in Rzhetsky et. al. 2007. In this case, the patient
            hospitalization rate is assumed to be the identical for every
            disease. This constant hospitalization rate is set so
            that the prevalence of Schizophrenia patients in the EMR
            dataset matches the prevalence of Schizophrenia in the general
            population  (1.1%).

        (2) "None": No patient counts normalization is performed.

        (3) "max": Hospitalization rates are allowed to be different for
            each individual disease. This allow the prevalence of the
            each disease in the EMR dataset to be independently adjusted to
            match the prevalence of that disease in the general population.
            (see data/disease-prevalence.csv). Furthermore, the prevalence
            of each disease-pair (D1 and D2) is adjusted so that the 
            hospitalization rates of patients with both D1 and D2 is the
            maximum of the hospitalization rate of patients with D1 and
            hospitalization rate of patients with D2.

        (4) "min": Same is (3) but replace "maximum" with "minimum"

        (5) "avg": Same as (3) but replace "maximum" with "average"

        (6) "wts": Same as (3) but replace "maximum" with "weighted"
            average". The D1 and D2 hospitalization rate are weighted,
            respectively, by the counts of patients with D1 and counts of 
            patients with D2, 

        (7) "sum" Same as (30 but replace "maximum" with sum".

    We are currently investigating if normalization schemes (3), (4), (5) (6),
    and (7) can potentially reduce the effect of Berkson's bias [2,3].

EXAMPLES

    # Use all default parameters
    python p_value.py -i data/test_EMR_dataset.csv

    # Select disease pairs.
    python p_value.py -i data/test_EMR_dataset.csv \
                      --d1 "Breast cancer (female)" --d2 "Epilepsy"

    python p_value.py -i data/test_EMR_dataset.csv \
                      --d1 "Attention deficit" --d2 "Epilepsy"

    # Select tau1 and tau2 values 
    python p_value.py -i data/test_EMR_dataset.csv --tau1 6 --tau2 6

    # Select pravalence normalization method
    python p_value.py -i data/test_EMR_dataset.csv --norm_prval_method avg

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>

REFERENCES:

    [1] Rzhetsky, A., Wajngurt, D., Park, N. & Zheng, T. Probing genetic 
        overlap among complex human phenotypes. Proc Natl Acad Sci U S A 
        104, 11694-9 (2007).

    [2] Berkson, J. Limitations of the application of fourfold table analysis
        to hospital data. Biometrics 2, 47-53 (1946).

    [3] http://en.wikipedia.org/wiki/Berkson's_paradox
"""

import sys
import time
import optparse

from scipy.stats import chi2

from emr_database_class import EMRDatabase
from optimize_log_likelihood_class import OptimizeLogLikelihood

def main():

    global options

    # Import EMR data into database
    database = EMRDatabase()

    database.import_data(options.emr_data_file,
                         options.diseases_file,
                         options.code2disease_file)


    # Instantiate the OptimizeLogLikelihood class
    opt_log_likelihood = OptimizeLogLikelihood(options.verbose)
    opt_log_likelihood.set_use_random_seed(options.use_random_seed)

    f = open("p_value.txt", 'w')
    f.write("# emr_data_file = %s\n" % options.emr_data_file)
    f.write("# norm_prval_method = %s\n" % options.norm_prval_method)
    f.write("# threshold_type = %s\n" % options.threshold_type)
    f.write("\n")
    f.write("# D1, D2, overlap_type, ")
    f.write("overlap_LL, indepedent_LL, ")
    f.write("LLR, p_value\n")

    # Compute p-value for specified disease pair or for all disease pairs with 
    # non-zero patient counts.
    if options.disease1 != None and options.disease2 != None:
        D1_list = [options.disease1]
        D2_list = [options.disease2]
    else:
        diseases = database.get_diseases()
        diseases.sort()
        filtered_diseases = __filter_diseases(diseases, database)
        D1_list = filtered_diseases
        D2_list = filtered_diseases

    for D1 in D1_list:
        for D2 in D2_list:

            if options.disease1 == None or options.disease2 == None:
                if D1 >= D2: continue

            print "-" * 80
            print "D1= %s, D2= %s," % (D1, D2),
            print "overlap_type= %s" % options.overlap_type

            # Independent (no genetic overlap) model
            indep_log_likelihood = (
                __compute_log_likelihood_wrapper(opt_log_likelihood,
                                                 database,
                                                 D1, D2,
                                                 options.tau1,
                                                 options.tau2,
                                                 "independent",
                                                 options.threshold_type,
                                                 options.prevalence_file,
                                                 options.norm_prval_method))

            min_log_likelihood = indep_log_likelihood - 1.0

            # Allow genetic overlap model
            overlap_log_likelihood = (
                __compute_log_likelihood_wrapper(opt_log_likelihood,
                                                 database,
                                                 D1, D2,
                                                 options.tau1,
                                                 options.tau2,
                                                 options.overlap_type,
                                                 options.threshold_type,
                                                 options.prevalence_file,
                                                 options.norm_prval_method,
                                                 min_log_likelihood))


            log_likelihood_ratio = 2.0 * (overlap_log_likelihood -
                                          indep_log_likelihood)

            # Degree of freedoms of the chi-square distribution.
            dof = 1  

            # p-value is the area at the right tail of the chi-square
            # distribution
            p_value = 1.0 - chi2.cdf(log_likelihood_ratio, dof)

            text = "%s, %s, %s, " % (D1, D2, options.overlap_type)
            text += "%.3E, " % overlap_log_likelihood
            text += "%.3E, " % indep_log_likelihood
            text += "%.3E, %.3E" % (log_likelihood_ratio, p_value)

            print "overlap_LL= %.2E," % overlap_log_likelihood,
            print "indep_LL= %.2E," % indep_log_likelihood,
            print "LLR= %.2E," % log_likelihood_ratio,
            print "p_value= %.2E" % p_value
            print "-" * 80
            print


            f.write(text + '\n')

    f.close()

def __compute_log_likelihood_wrapper(opt_log_likelihood,
                                     database,
                                     D1, D2,
                                     tau1, tau2,
                                     overlap_type,
                                     threshold_type,
                                     prevalence_file,
                                     norm_prval_method,
                                     min_log_likelihood=None):
    """Setup log_likelihood function and then compute the optimized log 
    likelihood value.

    Notes
    -----
    Rerun optimization until the optimal log_likelihood is greater
    than the specified min_log_likelihood value.
    """

    opt_log_likelihood.setup_log_likelihood_func(database,
                                                 D1, D2,
                                                 tau1, tau2,
                                                 overlap_type,
                                                 threshold_type,
                                                 prevalence_file,
                                                 norm_prval_method)

    while True:

        log_likelihood, _, _ = opt_log_likelihood.run()

        if min_log_likelihood == None: break

        if log_likelihood > min_log_likelihood: break

    return log_likelihood

def __filter_diseases(diseases, database):
    """Filter for diseases where patients counts for disease is non-zero."""

    filtered_Ds = []

    for D in diseases:

        D_patients = database.query_emr_data(D)
        if len(D_patients) != 0: filtered_Ds.append(D)

    return filtered_Ds

if __name__ == '__main__':

    start_time = time.time()

    usage = """python p_value.py -i <emr_data_file> --d1 <1st_disease_name> 
            --d2 <2nd_disease_name>"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data", action="store", type="string",
                      default=None,
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

    parser.add_option("--overlap_type",  action="store",  type="string",
                      default="cooperation", dest="overlap_type",
                      help="type of genetic overlap used in the model.")

    parser.add_option("--threshold_type", action="store", type="string",
                      default="sharp", dest="threshold_type", 
                      help="type of genetic penetrance used in the model.")

    parser.add_option("--use_random_seed", action="store_true", default=False,
                      dest="use_random_seed",
                      help="random seed for pseudo-random number generator")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=False,
                      dest="verbose", help="no verbose output.")

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if not options.emr_data_file:
        parser.error("option -i required")

    if options.norm_prval_method == "None":
        options.norm_prval_method = None

    if options.norm_prval_method not in [None, "Rzhetsky", "max", "min", "avg",
                                         "wts_avg", "sum"]:

        raise ValueError("invalid --norm_prval_method %s " %
                         options.norm_prval_method)

    if options.overlap_type not in ["cooperation",
                                    "competition"]:
        raise ValueError("invalid --overlap_type %s " % options.overlap_type)

    if options.threshold_type not in ["sharp", "soft"]:
        raise ValueError("invalid --threshold_type %s " %
                         options.threshold_type)

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
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
