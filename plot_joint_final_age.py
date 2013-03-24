#!/usr/bin/env python
"""
SYNOPSIS
    This script plots the joint patient's final-age distribution for a pair of
    diseases using the data derived from the EMR dataset.

DESCRIPTION
    For each pair of diseases D1 and D2, there are four possible phenotype
    status = {PHI0, PHI1, PHI2, PHI12} corresponding to
        PHI0 = "affected by neither D1 or D2"
        PHI1 = "affected by D1 but not D2"
        PHI2 = "affected by D2 but not D1"
        PHI12 = "affected by both D1 and D2"

    The patients are seperated by their 2 disease phenotype status and the
    counts distribution are plotted seperately for each phenotype status.

EXAMPLES

    # Plot for specified disease pair.
    python plot_joint_final_age.py -i data/test_EMR_dataset.csv \
        --d1 "Breast cancer (female)" --d2 "Schizophrenia"

    # Plot for all disease pairs with non-zero patient counts.
    python plot_joint_final_age.py -i data/test_EMR_dataset.csv

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt

from emr_database_class import EMRDatabase
from joint_final_age_class import JointFinalAge

def main():

    global options

    database = EMRDatabase()

    database.import_data(options.emr_data_file,
                         options.diseases_file,
                         options.code2disease_file)

    diseases = database.get_diseases()
    diseases.sort()

    # Create plot for specified disease pair or for all disease pairs with 
    # non-zero patient counts.
    if options.disease1 != None and options.disease2 != None:

        patient_counts, final_age_array = __get_counts(database,
                                                       options.disease1,
                                                       options.disease2,
                                                       options.verbose)

        plt = __plot_figure(final_age_array, patient_counts,
                            options.disease1, options.disease2,
                            options.normalize, options.verbose)
        plt.show()

    else:  

        filtered_diseases = __filter_diseases(diseases, database)

        for D1 in filtered_diseases:
            for D2 in filtered_diseases:

                if D1 == D2: continue

                patient_counts, final_age_array = __get_counts(database,
                                                               D1, D2,
                                                               options.verbose)

                plt = __plot_figure(final_age_array, patient_counts, D1, D2,
                                    options.normalize, options.verbose)
                plt.close()

def __filter_diseases(diseases, database):
    """Filter for diseases where patients count for disease is non-zero."""

    filtered_Ds = []

    for D in diseases:

        D_patients = database.query_emr_data(D)
        if len(D_patients) != 0: filtered_Ds.append(D)

    return filtered_Ds

def __get_counts(database, D1, D2, verbose):
    """Extract counts of patients as a distribution of the final age."""

    # Patients with D1, D2 or both.
    D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True)

    # Patients without D1 and without D2.
    noD1D2_patients = database.query_emr_data(["not " + D1, "not " + D2])

    # Extract counts of patients as a distribution of the final
    # age value, seperately for each of 4 disease phenotypes.
    joint_final_age = JointFinalAge(D1orD2_patients, noD1D2_patients, verbose)

    final_age_array = joint_final_age.get_final_age_array()

    patient_counts = joint_final_age.get_patient_counts()

    return (patient_counts, final_age_array)


def __plot_figure(final_age_array, patient_counts, D1, D2, normalize, verbose):
    """Plot and save the figure with pyplot."""

    sum_counts = np.sum(patient_counts, axis=1) 

    if normalize:
        for index, sum_count in enumerate(sum_counts):
            patient_counts[index] = patient_counts[index] / sum_count

    plt.clf()  # clear the current figure.
    plt.plot(final_age_array, patient_counts[0],'mx') #  PHI0
    plt.plot(final_age_array, patient_counts[1],'rs') #  PHI1
    plt.plot(final_age_array, patient_counts[2],'g^') #  PHI2
    plt.plot(final_age_array, patient_counts[3],'bo') #  PHI12

    plt.title('Patients Distribution for %s and %s' % (D1, D2),
              fontsize='large')

    y_max = 1.4* max(np.max(patient_counts[0]), np.max(patient_counts[1]),
                     np.max(patient_counts[2]), np.max(patient_counts[3]))

    plt.xlim([0.00, 100.00])
    plt.ylim([0.00, y_max])
    plt.xlabel('Patient\'s Final Age', fontsize='large')

    if normalize:
        plt.ylabel('Normalized Patient Counts', fontsize='large')
    else:
        plt.ylabel('Raw Patient Counts', fontsize='large')

    plt.legend(['no %s and no %s (total= %d)' % (D1, D2, sum_counts[0]),
                'only %s (total= %d)' % (D1, sum_counts[1]),
                'only %s (total= %d)' % (D2, sum_counts[2]),
                'both %s and %s (total= %d)' % (D1, D2, sum_counts[3])],
                loc='best',
                prop={'size':10})

    # Save the plot as a png image.
    fig_name = "joint_final_age_%s_and_%s.png" % (D1, D2)
    fig_name = fig_name.replace(" ", "_")
    if verbose:
        print "saving %s to file" % fig_name
    plt.savefig(fig_name, format='png') 

    return plt

if __name__ == '__main__':

    start_time = time.time()

    usage = """python plot_joint_final_age.py -i <emr_data_file> 
            --d1 <1st_disease_name> --d2 <2nd_disease_name>"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data",
                      action="store", type="string", dest="emr_data_file",
                      help="RAW EMR data file.")

    parser.add_option("--d1", default=None, action="store", type="string",
                      dest="disease1", 
                      help="name of the first disease.")

    parser.add_option("--d2", default=None, action="store", type="string",
                      dest="disease2", 
                      help="name of the second disease.")

    parser.add_option("--no_normalize", action="store_false", default=True,
                      dest="normalize", 
                      help="don't normalize by the total counts.")

    parser.add_option("--diseases_file", action="store",
                      default="data/disease-list.txt", type="string",
                      dest="diseases_file",
                      help="file contain list of diseases.")

    parser.add_option("--code2disease_file", action="store",
                      default="data/code-to-disease.csv", type="string",
                      dest="code2disease_file",
                      help="CSV file mapping ICD9 code to disease.")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=False,
                      dest="verbose", help="no verbose output.")

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if not options.emr_data_file:
        parser.error("option -i required")

    if options.verbose:
        print "-" * 50
        print time.asctime()
        print "disease1: %s" % options.disease1
        print "disease2: %s" % options.disease2
        print "normalize: %s" % options.normalize
        print "emr_data_file: %s" % options.emr_data_file
        print "diseases_file: %s" % options.diseases_file
        print "code2disease_file: %s" % options.code2disease_file
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
