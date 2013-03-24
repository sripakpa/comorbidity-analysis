#!/usr/bin/env python
"""
SYNOPSIS
    This script plots the joint age-of-onset distribution for a pair of 
    diseases using the data derived from the EMR dataset.

DESCRIPTION
    The age-of-onset curve is the cumulative probability that a patient first 
    manifest the disease D1 at or before age T = t  given that the patient
    does eventually manifest the disease in his/her life time (T != None).

    For a pair of diseases D1 and D2, the corresponding age-of-onset curve 
    A1(t) and A2(t) can be used to link the conditional probabilities of the
    age-t phenotype [phi(t)] given the ultimate (age-integrated) phenotype 
    [phi(inf)] for all 4 possible two disease phenotype status.

    For each pair of diseases D1 and D2, there are four possible phenotype
    status = {PHI0, PHI1, PHI2, PHI12} corresponding to
        PHI0 = "affected by neither D1 or D2"
        PHI1 = "affected by D1 but not D2"
        PHI2 = "affected by D2 but not D1"
        PHI12 = "affected by both D1 and D2"

    Both phi(t) and phi(inf) can be any of the four phenotype status, so for
    each current age T = t, there are 16 conditional probability to compute
    in total.

    P(phi(t) | phi(inf)) table:

                                  phi(inf)
    --------------------------------------------------------------------------
                PHI0        PHI1         PHI2        PHI12
    --------------------------------------------------------------------------
    phi(t)
    PHI0        1           1 - A1(t)    1 - A2(t)   (1 - A1(t)) * (1 - A2(t))
    PHI1        0           A1(t)        0           A1(t) * (1 - A2(t))
    PHI2        0           0            A2(t)       (1 - A1(t)) * A2(t)
    PHI12       0           0            0           A1(t) * A2(t)
    --------------------------------------------------------------------------
    Total       1           1            1           1
    --------------------------------------------------------------------------

    This scripts plots the 16 conditional probabilities curve 
    as a 4 x 4 subplots on a single multiplots figure.

EXAMPLES
    # Plot for specified disease pair.
    python plot_joint_age_of_onset.py \
        -i data/test_EMR_dataset.csv \
        --d1 "Breast cancer (female)" --d2 "Schizophrenia"

    # Plot for all disease pairs with non-zero patient counts.
    python plot_joint_age_of_onset.py \
        -i data/test_EMR_dataset.csv

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt

from emr_database_class import EMRDatabase
from joint_age_of_onset_class import JointAgeOfOnset


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

        funcs = __get_funcs(database,
                            options.disease1,
                            options.disease2,
                            options.verbose)

        plt = __plot_figure(funcs,
                            options.disease1,
                            options.disease2,
                            options.verbose)

        plt.show()

    else:

        filtered_diseases = __filter_diseases(diseases, database)

        for D1 in filtered_diseases:
            for D2 in filtered_diseases:

                if D1 == D2: continue

                funcs = __get_funcs(database,
                                    D1, D2,
                                    options.verbose)

                plt = __plot_figure(funcs,
                                    D1, D2,
                                    options.verbose)

                plt.close()  # close current figure

def __filter_diseases(diseases, database):
    """Filter for diseases where patients counts for disease is non-zero."""

    filtered_Ds = []

    for D in diseases:

        D_patients = database.query_emr_data(D)
        if len(D_patients) != 0: filtered_Ds.append(D)

    return filtered_Ds

def __get_funcs(database, D1, D2, verbose):
    """Empirically construct the conditional probabilities P(phi(t) | phi(inf))

    Returns
    -------
    funcs: 4 x 4 numpy array of scipy interp1d objects.
          Each joint_age_of_onsets[i,j] is a scipy 1-D interpolated function
          object representing the P(phi(t) | phi(inf)) conditional
          probabilities.
          Index j is the index of the phi(t) phenotype.
          Index j is the index of the phi(inf) phenotype.
          Possible value of index i and index j are:
              0 : PHI0, "affected by neither D1 or D2"
              1 : PHI1 = "affected by D1 but not D2"
              2 : PHI2 = "affected by D2 but not D1"
              3 : PHI12 = "affected by both D1 and D2"
    """

    # Patients that have D1, D2 or both
    D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True)

    joint_age_of_onset = JointAgeOfOnset(D1orD2_patients, verbose)

    funcs = joint_age_of_onset.get_funcs()

    return funcs

def __plot_figure(funcs, D1, D2, verbose):
    """Plot and save the 4 x 4 joint age of onsets curve with pyplot."""

    num_rows = funcs.shape[0]
    num_cols = funcs.shape[1]

    if num_rows != 4: raise ValueError("num_rows must be 4.")

    if num_cols != 4: raise ValueError("num_cols must be 4.")

    Phi_label = [r'$\Phi_0$', r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_{12}$']
    indices = [(i,j) for i in xrange(num_rows) for j in xrange(num_cols)]

    #define plot size in inches (width, height) & resolution(DPI)
    fig = plt.figure(figsize=(8.0, 6.0), dpi=100)

    # axes coordinates are 0,0 is bottom left and 1,1 is upper right
    axes = fig.add_axes([0,0,1,1])

    plt.clf()

     # shift plot position to make room for labels.
    fig.subplots_adjust(left=0.18, right=0.97)
    fig.subplots_adjust(bottom=0.09, top=0.875)

    plt.rc("font", size=10)  # define font size
    plt.rc('lines', linewidth=4)  # set line width

    for (row_id, col_id) in indices:

        # plot_num = 1 is the first plot number and increasing plot_num fill 
        # rows first. max(plotNum) == num_rows * num_cols.
        plot_num = (col_id + 1) + num_cols * row_id

        subplot = plt.subplot(num_rows, num_cols, plot_num)

        xnew = np.linspace(0, 200, num=2000)

        plt.plot(xnew, funcs[row_id, col_id](xnew), 'b-')

        subplot.set_xticks([0,50,100])
        subplot.set_xlim([0.00, 100.00])

        subplot.set_yticks([0.00,0.50,1.00])
        subplot.set_ylim([-0.10, 1.10])

        if row_id == 0:
            subplot.set_xticklabels([]) # hide xtick 
            subplot.set_xlabel(r'$\phi(\infty)$=' + Phi_label[col_id],
                               fontsize=12)
            subplot.xaxis.set_label_position("top")
        elif row_id == num_rows - 1:
            pass # don't remove xlabel
        else: 
            # hide xtick and xlabel
            subplot.set_xlabel('') 
            subplot.set_xticklabels([])

        if col_id == 0:
            subplot.set_ylabel(r'$\phi(t)$=' + Phi_label[row_id],
                               rotation='horizontal',
                               fontsize=12)
        else:
            # hide ytick and ylabel
            subplot.set_ylabel('') 
            subplot.set_yticklabels([])

    # Global title.
    plt.text(0.96, 0.98,
             'Joint Age-of-onset Distribution for %s and %s' % (D1, D2),
             horizontalalignment='right', verticalalignment='top',
              fontsize=12, transform = axes.transAxes)

    # Global xlabel.
    plt.text(0.575, 0.02,
             't (age)',
             horizontalalignment='center', verticalalignment='bottom',
             fontsize=12, transform = axes.transAxes)

    # Global ylabel.
    plt.text(0.02, 0.5, 
             'Probability[ ' + r'$\phi(t)$' + ' | ' + r'$\phi(\infty)$' +' ]',
             horizontalalignment='left', verticalalignment='center',
             fontsize=12, transform = axes.transAxes, rotation='vertical')

    # Save the plot as a png image.
    fig_name = "joint_age_of_onset_%s_and_%s.png" % (D1, D2)
    fig_name = fig_name.replace(" ", "_")

    if verbose: print "saving %s to file" % fig_name
    plt.savefig(fig_name, format='png')

    return plt

if __name__ == '__main__':

    start_time = time.time()

    usage = """python plot_joint_age_of_onsets.py -i <emr_data_file>
            --d1 <1st_disease_name> --d2 <2nd_disease_name>"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data",
                      action="store", type="string", dest="emr_data_file",
                      help="RAW EMR data file")

    parser.add_option("--d1", default=None, action="store", type="string",
                      dest="disease1", 
                      help="name of the first disease.")

    parser.add_option("--d2", default=None, action="store", type="string",
                      dest="disease2", 
                      help="name of the second disease.")

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
        print "disease1: %s" % options.disease1
        print "disease2: %s" % options.disease2
        print "emr_data_file: %s" % options.emr_data_file
        print "diseases_file: %s" % options.diseases_file
        print "code2disease_file: %s" % options.code2disease_file
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
