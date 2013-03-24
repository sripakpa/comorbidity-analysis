#!/usr/bin/env python
"""
SYNOPSIS
    This script plots the age-integrated phenotype probability for a pair of
    diseases D1 and D2 using the genetic penetrance models proposed by
    Rzhetsky et al. 2007.

DESCRIPTION
    For each pair of diseases D1 and D2, there are four possible phenotype
    status = {PHI0, PHI1, PHI2, PHI12} corresponding to
        PHI0 = "affected by neither D1 or D2"
        PHI1 = "affected by D1 but not D2"
        PHI2 = "affected by D2 but not D1"
        PHI12 = "affected by both D1 and D2"

    This script create contour plots (as a function of rho1 and rho2) for:

        phi_infty_probs[0] = P(phi(infty) = PHI0; rho1, rho2, rho12)
        phi_infty_probs[1] = P(phi(infty) = PHI1; rho1, rho2, rho12)
        phi_infty_probs[2] = P(phi(infty) = PHI2; rho1, rho2, rho12)
        phi_infty_probs[3] = P(phi(infty) = PHI12; rho1, rho2, rho12)

    User can also specified the value of other parameters in the model (tau1,
    tau2, overlap_type, theshold_type) from command line.

EXAMPLES

    # Plot using default model parameters
    python plot_genetic_penetrance.py

    # Plot with specified tau1 and tau2
    python plot_genetic_penetrance.py --tau1 1 --tau2 1

    # Plot with all parameters of the genetic penetrance model specified.
    python plot_genetic_penetrance.py --tau1 1 --tau2 1 --rho12 1.0 \
        --overlap_type "cooperation" --threshold_type "sharp"

    # Plot with all parameters of the genetic penetrance model specified.
    python plot_genetic_penetrance.py --tau1 1 --tau2 3 --rho12 0.0 \
        --overlap_type "independent" --threshold_type "sharp"

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt

from genetic_penetrance_class import GeneticPenetrance

def main():

    global options

    genetic_penetrance = GeneticPenetrance(options.tau1,
                                           options.tau2,
                                           options.overlap_type,
                                           options.threshold_type,
                                           options.verbose)

    plt = __plot_contour(genetic_penetrance,
                         options.rho12,
                         options.verbose)

    if options.show_plot: plt.show()

def __plot_contour(genetic_penetrance, rho12, verbose):
    """Plot and save the age-integrated phenotype probabilities contour
    with pyplot."""

    tau1 = genetic_penetrance.get_tau1()
    tau2 = genetic_penetrance.get_tau2()
    overlap_type = genetic_penetrance.get_overlap_type()
    threshold_type = genetic_penetrance.get_threshold_type()

    delta1 = 0.025 * tau1
    delta2 = 0.025 * tau2

    rho_min = 0
    rho_max = 3

    # tau1 and tau2 acts as a scaling factor of rho1 and rho2, respectively.
    rho1_array = np.arange(rho_min * tau1, rho_max * tau1, delta1) + delta1
    rho2_array = np.arange(rho_min * tau2, rho_max * tau2, delta2) + delta2

    # notice the switch in the order of rho1 and rho2 here!
    # if do not switch, then will get the tranpose of want we want!
    RHO2, RHO1 = np.meshgrid(rho2_array, rho1_array)

    PROBS = np.zeros([4, rho1_array.size, rho2_array.size], dtype=np.float64)

    for rho1_index, rho1 in enumerate(rho1_array):
        for rho2_index, rho2 in enumerate(rho2_array):

            PROBS[:,
                  rho1_index,
                  rho2_index] = genetic_penetrance.compute_probs(rho1,
                                                                 rho2,
                                                                 rho12)

    #define plot size in inches (width, height) & resolution(DPI)
    fig = plt.figure(figsize=(8.0, 6.0), dpi=100)

    # axes coordinates are 0,0 is bottom left and 1,1 is upper right
    axes = fig.add_axes([0,0,1,1])

    plt.clf()

    Phi_label = [r'$\Phi_0$', r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_{12}$']

    for index in [0, 1, 2, 3]:

        subplot = plt.subplot(2, 2, index + 1)
        CS = plt.contour(RHO1, RHO2, PROBS[index])
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel(r'$\rho_1$')
        plt.ylabel(r'$\rho_2$')

        subplot.set_xticks(np.arange(rho_min, rho_max + 1) * tau1)
        subplot.set_xlim([rho_min * tau1, rho_max * tau1])

        subplot.set_yticks(np.arange(rho_min, rho_max + 1) * tau2)
        subplot.set_ylim([rho_min * tau2, rho_max * tau2])

        plt.legend(['P(' + r'$\phi(\infty)$' + ') = %s' % Phi_label[index]],
                    loc='best',
                    prop={'size':10})


    # Global title.
    plt.text(0.50, 0.98,
             'P(' + r'$\phi(\infty)$;' +
              r'$\rho_1$' + ', '
              r'$\rho_2$' '), ' +
              r'$\rho_{12}$' + ' = %.1f, ' % rho12 +
              r'$\tau_1$' + ' = %d, ' % tau1 +
              r'$\tau_2$' + ' = %d, ' % tau2 +
              'type = (%s, %s)' % (overlap_type, threshold_type),
              horizontalalignment='center', verticalalignment='top',
              fontsize=12, transform = axes.transAxes)


    # Save the plot as a png image.

    fig_name = "genetic_penetrance_"
    fig_name += "%s_" % overlap_type
    fig_name += "rho12=%.1f_" % rho12
    fig_name += "tau1=%d_" % tau1
    fig_name += "tau2=%d" % tau2
    fig_name +=".png" 

    if verbose: print "saving %s to file" % fig_name

    plt.savefig(fig_name, format='png') 

    return plt


if __name__ == '__main__':

    start_time = time.time()

    usage = """python plot_genetic_penetrance.py"""

    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("--tau1", default=1, action="store", type="int",
                      dest="tau1", 
                      help="tau1 parameter of the genetic penetrance model.")

    parser.add_option("--tau2", default=1, action="store", type="int",
                      dest="tau2", 
                      help="tau2 parameter of the genetic penetrance model.")

    parser.add_option("--rho12", default=0.5, action="store", type="float",
                      dest="rho12", 
                      help="rho12 parameter of the genetic penetrance model.")

    parser.add_option("--overlap_type", default="cooperation", action="store", 
                      type="string", dest="overlap_type",
                      help="type of genetic overlap used in the model.")

    parser.add_option("--threshold_type", default="sharp", action="store", 
                      type="string", dest="threshold_type", 
                      help="type of genetic penetrance used in the model.")

    parser.add_option("--no_show_plot", action="store_false", default=True,
                      dest="show_plot",
                      help="don't show plot on screen.")

    parser.add_option("-v", "--verbose", action="store_true", default=True,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=True,
                      dest="verbose", help="verbose output.")

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if options.overlap_type not in ["cooperation",
                                    "competition",
                                    "independent"]:
        raise ValueError("invalid --overlap_type value specified")

    if options.threshold_type not in ["sharp", "soft"]:
        raise ValueError("invalid --threshold_type value specified")

    if options.verbose:
        print "-" * 50
        print time.asctime()
        print "tau1: %s" % options.tau1
        print "tau2: %s" % options.tau2
        print "rho12: %s" % options.rho12
        print "overlap_type: %s" % options.overlap_type
        print "threshold_type: %s" % options.threshold_type
        print "show_plot: %s" % options.show_plot
        print "-" * 50

    main()
    if options.verbose:
        print time.asctime(),
        print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
