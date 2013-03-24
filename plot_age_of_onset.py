#!/usr/bin/env python
"""
SYNOPSIS
    This script plots the age-of-onset distribution of diseases using the data
    derived from the electronic medical records (EMRs).

DESCRIPTION
    The age-of-onset distribution is defined as the cumulative probability
    that a disease/disease manifest itself before or at any given age,
    conditioned on the outcome that the patient does get the disease sometime
    in his or her lifespan.

EXAMPLES

    # Plot for single selected disease
    python plot_age_of_onset.py -i data/test_EMR_dataset.csv  \
        --disease "Attention deficit"

    # Plot for all disease in the diseases_file with non-zero patient counts.
    python plot_age_of_onset.py -i data/test_EMR_dataset.csv

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt

from emr_database_class import EMRDatabase
from age_of_onset_func import create_age_of_onset_func


def main():

    global options

    database = EMRDatabase()

    database.import_data(options.emr_data_file,
                         options.diseases_file,
                         options.code2disease_file)

    diseases = database.get_diseases()
    diseases.sort()

    # Create plot for the specified disease or all diseases with non-zero
    # patient counts.
    if options.disease != None:

        M_func, F_func, M_count, F_count = __get_funcs(database, 
                                                       options.disease)

        plt = __plot_figure(M_func, F_func, M_count, F_count,
                            options.disease, options.verbose)

        plt.show()

    else:

        filtered_diseases = __filter_diseases(diseases, database)

        for D in filtered_diseases:

            M_func, F_func, M_count, F_count = __get_funcs(database, D)

            plt = __plot_figure(M_func, F_func, M_count, F_count,
                                D, options.verbose)

            plt.close()

def __filter_diseases(diseases, database):
    """Filter for diseases where patients counts for disease is non-zero."""

    filtered_diseases = []

    for D in diseases:

        D_patients = database.query_emr_data(D)
        if len(D_patients) != 0: filtered_diseases.append(D)

    return filtered_diseases

def __get_funcs(database, D):
    """Empirically construct age-of-onset curve for the disease."""

    male_patients = database.query_emr_data(D, gender_filters = "M")
    female_patients = database.query_emr_data(D, gender_filters = "F")

    male_func , _, _ = create_age_of_onset_func(male_patients)
    female_func , _, _ = create_age_of_onset_func(female_patients)

    male_count = len(male_patients)
    female_count = len(female_patients)

    return male_func, female_func, male_count, female_count

def __plot_figure(M_func, F_func, M_count, F_count, D, verbose):
    """Plot and save the age of onset curve with pyplot."""

    age_array = np.linspace(0, 200, num=2000)

    plt.clf()  # clear the current figure.

    plt.plot(age_array, M_func(age_array),'b-')
    plt.plot(age_array, F_func(age_array),'r-')

    plt.title('Age-of-onset Distribution for %s' % D, fontsize='x-large')

    plt.ylabel('Cummulative Probability', fontsize='x-large')
    plt.ylim([0.00, 1.10])

    plt.xlabel('Age', fontsize='x-large')
    plt.xlim([0.00, 100.00])

    legend = plt.legend(['male (%d)' % M_count,
                         'female (%d)' % F_count], loc='best')

    # Save the plot as a png image.
    fig_name = D.replace(" ", "_") + '.png'

    if verbose: print "saving %s to file" % fig_name

    plt.savefig(fig_name, format='png')

    return plt

if __name__ == '__main__':

    start_time = time.time()

    usage = """python plot_age_of_onset.py -i <emr_data_file> 
            --disease <disease_name>"""
    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data",
                      action="store", type="string", dest="emr_data_file",
                      help="RAW EMR data file")

    parser.add_option("--disease", default=None, action="store", type="string",
                      dest="disease", 
                      help="select a single disease.")

    parser.add_option("--diseases_file", action="store",
                      default="data/disease-list.txt", type="string",
                      dest="diseases_file",
                      help="file contain list of diseases.")

    parser.add_option("--code2disease_file", action="store",
                      default="data/code-to-disease.csv", type="string",
                      dest="code2disease_file",
                      help="CSV file mapping ICD9 code to disease.")

    parser.add_option("-v", "--verbose", action="store_true", default=True,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=True,
                      dest="verbose", help="no verbose output.")

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if not options.emr_data_file:
        parser.error("option -i required")

    if options.verbose:
        print "-" * 50
        print time.asctime()
        print "disease: %s" % options.disease
        print "emr_data_file: %s" % options.emr_data_file
        print "diseases_file: %s" % options.diseases_file
        print "code2disease_file: %s" % options.code2disease_file
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
