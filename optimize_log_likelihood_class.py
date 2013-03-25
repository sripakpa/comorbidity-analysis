#!/usr/bin/env python

import math
import time

import numpy as np
from scipy import optimize

from genetic_penetrance_class import GeneticPenetrance
from joint_age_of_onset_class import JointAgeOfOnset
from joint_final_age_class import JointFinalAge
from norm_prevalence_func import create_prval_norm_func
from log_likelihood_func import create_log_likelihood_func
from log_likelihood_func import create_log_likelihood_fprime

class OptimizeLogLikelihood(object):
    """OptimizeLogLikelihood class

    Find the model's parameters (rho1, rho2 and rho12) that maximize the log-
    likelihood of observing the EMR data according to the genetic penetrance
    models proposed by Rzhetsky et al. 2007.


    Parameters
    ----------
    verbose : bool, optional
        Verbose output.

    opt_method : ['Nelder-Mead', 'Powell', 'CG', 'BFGS' (default)], optional
        Name of method use to perform the optimization.

    use_analytical_gradient : bool, optional
        Analytically compute the gradient when using gradient-based
        optimization methods (i.e. 'CG' and 'BFGS')

    Attributes
    ----------
    tau1 : integer
        Model's parameter indicating the minimum number of deleterious mutation
        in S1 or S12, the combined region of the genome that predisposes the
        polymorphisms' bearers to disease D1
        The tau2 values used in Rzhetsky et al. 2007 were tau1 = 1 or tau1 = 3.

    tau2 : integer
        Model's parameter indicating the minimum number of deleterious mutation
        in S2 or S12, the combined region of the genome that predisposes the
        polymorphisms' bearers to disease D2
        The tau2 values used in Rzhetsky et al. 2007 were tau2 = 1 or tau2 = 3.

    overlap_type: ["independent", "cooperation", "competition"]
        Model's parameter describing the type of genetic overlap used in the
        model

    rng: np.random object
        pseudo-random number generator.

    log_likelihood_func : python function
        A function to evaluate the log_likelihood value
        as a function of rho1, rho2 and rho12

    log_likelihood_fprime : python function
        A function to evaluate the derivative of the log-likelihod wrt
        to rho1, rho2 and rho12.

    """

    def __init__(self, verbose=False):
        """Instantiate variables in LogLikelihood class."""

        self.opt_method = "BFGS"
        self.use_analytical_gradient = False
        self.rng = np.random.RandomState(50)  # fixed seed
        self.verbose = verbose

        self.log_likelihood_func = None
        self.log_likelihood_fprime = None

    def set_opt_method(self, opt_method):
        """Set optimization method name attribute"""

        self.opt_method = opt_method

    def set_use_analytical_gradient(self, use_analytical_gradient):
        """Set use_analytical_gradient attribute"""

    def set_use_random_seed(self, use_random_seed):
        """Use fixed or random seed for pseudo-random number generator."""

        # re-seed the pseudo-random number generator
        if use_random_seed: self.rng.seed()

    # Public methods
    def get_log_likelihood_func(self):

        if self.log_likelihood_func == None:
            raise RuntimeError("log_likelihood_func not yet setup.")

        return self.log_likelihood_func

    def setup_log_likelihood_func(self,
                                  database,
                                  D1, D2,
                                  tau1, tau2,
                                  overlap_type,
                                  threshold_type,
                                  prevalence_file=None,
                                  norm_prval_method=None):
        """Setup the log_likelihood function.

        Parameters
        ----------
        database : EMRDatabase class object
            This class allow query of EMR data by diseases, gender, and 
            ethnicity citeria

        D1 : string
            Name of the first disease.

        D2 : string
            Name of the first disease.

        tau1 : integer
            Model's parameter indicating the minimum number of deleterious
            mutation in S1 or S12.

        tua2 : integer
            Model's parameter indicating the minimum number of deleterious
            mutation in S2 or S12.

        overlap_type : ["independent", "cooperation", "competition"]
            Model's parameter describing the type of genetic overlap used in
            the model

        threshold_type : ["sharp", "soft"]
            Model's parameter describing the type of genetic penetrance used in
            the model

        prevalence_file : string
            Location of the prevalence data file

        norm_prval_method : [None, "rzhetsky", "max", "min",
                             "avg", "wts_avg", "sum"]
            Protocol to normalize the prevalence data.
        """

        # Query patients data
        D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True)
        noD1D2_patients = database.query_emr_data(["not " + D1, "not " + D2])

        # Conditional probabilities: P(phi(t) | phi(infty))
        joint_age_of_onset = JointAgeOfOnset(D1orD2_patients)

        joint_age_of_onset_funcs = joint_age_of_onset.get_funcs()

        # Patient counts of as a distribution of the final age value.
        joint_final_age = JointFinalAge(D1orD2_patients, noD1D2_patients)

        final_age_array = joint_final_age.get_final_age_array()
        patient_counts = joint_final_age.get_patient_counts()

        # Setup the genetic_penetrance model to compute the age-integrated
        # phenotype probabilities: P(phi(infty) ; rho1, rho2, rho12)
        genetic_penetrance = GeneticPenetrance(tau1,
                                               tau2,
                                               overlap_type,
                                               threshold_type)

        # Create a function that will normalize the raw EMR data to match the
        # general population disease prevalence.
        if prevalence_file != None and norm_prval_method != None:

            prval_norm_func = (
                create_prval_norm_func(D1, D2,
                                       database.get_disease2count(),
                                       database.get_tot_patient_count(),
                                       prevalence_file,
                                       norm_prval_method))
        else:
            prval_norm_func = None

        # Function to evaluate log-likelihood at given rho1, rho2 and rho12.
        log_likelihood_func = (
            create_log_likelihood_func(genetic_penetrance,
                                       joint_age_of_onset_funcs,
                                       patient_counts,
                                       final_age_array,
                                       prval_norm_func))

        # Function to evaluate the derivative of the log-likelihod wrt
        # to rho1, rho2 and rho12.
        log_likelihood_fprime = (
            create_log_likelihood_fprime(genetic_penetrance,
                                         joint_age_of_onset_funcs,
                                         patient_counts,
                                         final_age_array,
                                         prval_norm_func))

        self.tau1 = tau1
        self.tau2 = tau2
        self.overlap_type = overlap_type
        self.log_likelihood_func = log_likelihood_func
        self.log_likelihood_fprime = log_likelihood_fprime

    def run(self, save_path=False):
        """Optimize the log_likelihood function starting from a random points
        in the parameter space.

        Parameters
        ----------
        save_path: bool, optional
            save and return the coordinates along the optimization path.

        Returns
        -------
        opt_log_likelihood: float
            The log_likelihood value after optimization.

        opt_param: a dictionary
            Dictionary containing the coordinates after optimization.
            The key-value pairs are:
                "rho1" : value of rho1 (float)
                "rho2" : value of rho2 (float)
                "rho12" : value of rho12 (float)
                "log_likelihood" : log_likelihood value (float)

        paths : None or dictionary
            If save_path is False, then path is None.
            If save_path is True, then path is a dictionary storing the
            coordinates at each iteration along the optimization path. The
            key-value pairs are:

                "rho1" : list of rho1 values.
                "rho2" : list of rho2 values. 
                "rho12" : list of rho12 values. 
                "log_likelihood" : list of log_likelihood value.

                The ith element of each list store the the value at the ith
                optimization iteration step.
        """

        if self.log_likelihood_func == None:
            raise RuntimeError("log_likelihood_func not yet setup.")

        if self.log_likelihood_fprime == None:
            raise RuntimeError("log_likelihood_fprime not yet setup.")

        rho1_max = 1.0 * self.tau1
        rho2_max = 1.0 * self.tau2
        rho12_max = (rho1_max + rho2_max) / 2.0

        # uniform distribution over (0, 1]
        (rho1, rho2, rho12) = self.rng.rand(3) + 1.0e-20

        (rho1, rho2, rho12) = (rho1 * rho1_max,
                               rho2 * rho2_max,
                               rho12 * rho12_max)

        if self.verbose:
            print "Optimizing log_likelihood",
            print "(tau1 = %d, tau2 = %d," % (self.tau1, self.tau2),
            print "overlap_type = %s)" % self.overlap_type
            print "start: rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (rho1,
                                                                     rho2,
                                                                     rho12)

        # The log_likelihood func is not defined for negative rhos. Therefore,
        # optimize in the log_rho space instead of the rho space, since this
        # ensure that the rhos themselves never become negative.

        log_rho = True
        log_rho1 = math.log(rho1)
        log_rho2 = math.log(rho2)
        log_rho12 = math.log(rho12)

        log_likelihood_fprime = None
        if self.use_analytical_gradient:
            log_likelihood_fprime = self.log_likelihood_fprime

            if self.verbose:
                grad_error = (
                    optimize.check_grad(self.log_likelihood_func,
                                        log_likelihood_fprime,
                                        [log_rho1, log_rho2, log_rho12]))

                print "gradient calc error= " , grad_error

        minimize = self.__get_minimize_func()
        sign = -1.0 # maximize instead of minimize.

        start_time = time.time()

        xopt, allvec = minimize(self.log_likelihood_func,
                                [log_rho1, log_rho2, log_rho12],
                                args=(log_rho, sign,),
                                fprime=log_likelihood_fprime)

        if self.verbose:
            print " " * 8,
            print 'minimize_time = %.3f secs' % (time.time() - start_time)
            print " " * 8,
            print "rho1 = %.3f, rho2 = %.3f, rho12 = %.3f" % (np.exp(xopt[0]),
                                                              np.exp(xopt[1]),
                                                              np.exp(xopt[2]))

            for n, point in enumerate(allvec):
                # convert log_rhos back to rho 
                rho1 = np.exp(point[0])
                rho2 = np.exp(point[1])
                rho12 = np.exp(point[2])
                if self.overlap_type == "independent": rho12 = 0.0

                print " " * 8,
                print "%3d: rho1, rho2, rho12, log_likelihood = " % n,
                print "%.3f %.3f %.3f" % (rho1, rho2, rho12),
                print "%.3f" % self.log_likelihood_func(point)
            print

        # convert log_rho back to rho
        opt_rho1 = np.exp(xopt[0])
        opt_rho2 = np.exp(xopt[1])
        opt_rho12 = np.exp(xopt[2])
        if self.overlap_type == "independent": opt_rho12 = 0.0

        opt_param = dict()
        opt_param["rho1"] = opt_rho1
        opt_param["rho2"] = opt_rho2
        opt_param["rho12"] = opt_rho12
        opt_param["log_likelihood"] = self.log_likelihood_func(xopt)
        
        if save_path:
            path = {'rho1' : [], 'rho2': [], 'rho12': [], 'log_likelihood': []}
            for n, point in enumerate(allvec):
                # convert log_rhos back to rho 
                rho1 = np.exp(point[0])
                rho2 = np.exp(point[1])
                rho12 = np.exp(point[2])
                if self.overlap_type == "independent": rho12 = 0.0

                path['rho1'].append(rho1)
                path['rho2'].append(rho2)
                path['rho12'].append(rho12)
                path['log_likelihood'].append(self.log_likelihood_func(point))
        else:
            path = None

        opt_log_likelihood = opt_param["log_likelihood"]

        return opt_log_likelihood, opt_param, path

    # Private methods
    def __get_minimize_func(self):
        """Select and return the specified optimization function.

        Notes
        ------
        Use this function to experiment with different values of gtol and ftol

        gtol: float (scipy defualt is 1e-05)
            Gradient tolerance, stop when norm of gradient is less than gtol.
        ftol : float (scipy defualt is 0.0001)
            Function tolerance, relative error in func(xopt) acceptable for 
            convergence.
        """

        gtol_test = 0.001
        ftol_test = 0.001

        if self.opt_method == 'Nelder-Mead':
            # Nelder-Mead is not a gradient-based method

            def minimize(f, x0, args=(), fprime=None):
                return optimize.fmin(f, x0, args=args, ftol=ftol_test,
                                     retall=True, disp=self.verbose)
        elif self.opt_method == 'Powell':
            # Powell is not a gradient-based method

            def minimize(f, x0, args=(), fprime=None):
                return optimize.fmin_powell(f, x0, args=args, ftol=ftol_test, 
                                            retall=True, disp=self.verbose)
        elif self.opt_method == 'CG':
            # CG is a gradient-based method

            def minimize(f, x0, args=(), fprime=None):
                return optimize.fmin_cg(f, x0, args=args, gtol=gtol_test,
                                        retall=True, disp=self.verbose,
                                        fprime=fprime)

        elif self.opt_method == 'BFGS':
            # BFGS is a gradient-based descent method

            def minimize(f, x0, args=(), fprime=None):
                return optimize.fmin_bfgs(f, x0, args=args, gtol=gtol_test,
                                          retall=True, disp=self.verbose,
                                          fprime=fprime)

        else:
            raise ValueError("Unvalid opt_method %s" % opt_method)

        return minimize
