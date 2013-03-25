#!/usr/bin/env python
"""Create a function to compute the log_likelihood of observing the EMR 
patients data based on the genetic pentrance model proposed by Rzhetsky et al.
2007."""

import copy

import numpy as np


def create_log_likelihood_func(genetic_penetrance,
                               joint_age_of_onset_funcs,
                               patient_counts,
                               final_age_array,
                               prval_norm_func = None):
    """Create and return a function to evaluate the log_likelihood value
    as a function of rho1, rho2 and rho12."""

    def log_likelihood_func((rho1, rho2, rho12), log_rho=True, sign=1.0):
        """Define the log_likelihood function.

        Notes
        -----
        If log_rho=True, then input rhos are actually logarithm of the rhos
        and hence need to take exponent of them to get the actual rhos.

        The log_likelihood func is not defined for negative rhos. Therefore,
        advantagous to optimize in the log_rho space instead of the rho space,
        since this ensure that the rhos themselves never become negative.

        sign should be 1.0 to maximize the function.
        sign should be -1.0 to maximize the function.
        """

        if log_rho:
            rho1 = np.exp(rho1)
            rho2 = np.exp(rho2)
            rho12 = np.exp(rho12)

        log_likelihood = __compute_log_likelihood(rho1, rho2, rho12,
                                                  genetic_penetrance,
                                                  joint_age_of_onset_funcs,
                                                  patient_counts,
                                                  final_age_array,
                                                  prval_norm_func)

        # print "(rho1, rho2, rho12, log_likelihood) = %.5f %.5f %.5f %.5f" % (
        #        (rho1, rho2, rho12, log_likelihood) )

        return sign * log_likelihood

    return log_likelihood_func

def create_log_likelihood_fprime(genetic_penetrance,
                                 joint_age_of_onset_funcs,
                                 patient_counts,
                                 final_age_array,
                                 prval_norm_func = None):
    """Create and return a function to evaluate the deriviatve log_likelihood
    value as a function of rho1, rho2 and rho12."""

    def log_likelihood_fprime((rho1, rho2, rho12), log_rho=True, sign=1.0):
        """Define the log_likelihood derivative function.

        Notes
        -----
        If log_rho=True, then input rhos are actually logarithm of the rhos
        and hence need to take exponent of them to get the actual rhos.

        sign should be 1.0 to maximize the function.
        sign should be -1.0 to maximize the function.
        """

        if log_rho:
            # To avoid the sigularity at rho = 0.0
            rho1 = np.exp(rho1)
            rho2 = np.exp(rho2)
            rho12 = np.exp(rho12)

        deriv_log_likelihood = (
            __compute_deriv_log_likelihood(rho1, rho2, rho12,
                                           genetic_penetrance,
                                           joint_age_of_onset_funcs,
                                           patient_counts,
                                           final_age_array,
                                           prval_norm_func))

        # print "(rho1, rho2, rho12, deriv_log_likelihood) = ",
        # print "%.5f %.5f %.5f " % (rho1, rho2, rho12),
        # print deriv_log_likelihood

        return sign * deriv_log_likelihood

    return log_likelihood_fprime


def __compute_log_likelihood(rho1, rho2, rho12,
                             genetic_penetrance,
                             joint_age_of_onset_funcs,
                             patient_counts,
                             final_age_array,
                             prval_norm_func = None):
    """Compute the log likelihood of the genetic pentrance model given
    the observed EMR data.

    i.e. log-likihood( rho1, rho2 ; EMR_data )

    The genetic penetrance model is from Rzhetsky et al. 2007. For details
    of the likelihood computation, see  see Eq. 40 and Eq. 51 in SI Appendix 2.

    Parameters
    ----------
    rho1: float
        Parameter in the genetic penetrance model representing the expected
        number of deleterious polymorphisms in S1, the region of the genome
        that predisposes the polymorphisms' bearers to disease D1

    rho2: float
        Parameter in the genetic penetrance model representing the expected
        number of deleterious polymorphisms in S2, the region of the genome
        that predisposes the polymorphisms' bearers to disease D2

    rho12: float
        Parameter in the genetic penetrance model representing the expected
        number of deleterious polymorphisms in S12, the region of the genome
        that predisposes the polymorphisms' bearers to both disease D1 and 
        disease D2
        This parameter is set to zero in the independence model.

    genetic_penetrance: class object
        A class object with the method compute_probs(rho1, rho2, rho12)
        which returns a phi_infinity_probs array:

        e.g.: phi_infinity_probs = genetic_penetrance.compute_probs(rho1,
                                                                    rho2,
                                                                    rho12)

        phi_infinity_probs : 1D float numpy array (size = 4)
            The age-integrated phenotype for all 4 possible phenotype 2 
            diseases phenotype status:
                phi_infty_probs[0] = P(phi(infty) = PHI0; rho1, rho2, rho12)
                phi_infty_probs[1] = P(phi(infty) = PHI1; rho1, rho2, rho12)
                phi_infty_probs[2] = P(phi(infty) = PHI2; rho1, rho2, rho12)
                phi_infty_probs[3] = P(phi(infty) = PHI12; rho1, rho2, rho12)

    joint_age_of_onset_funcs : 4x4 numpy array of scipy interp1d objects.
        Each joint_age_of_onsets[i,j] is a scipy 1-D interpolated function
        object representing the P(phi(t) | phi(inf)) conditional probabilities.
        Index i is the index of the phi(t) phenotype.
        Index j is the index of the phi(inf) phenotype.
        Possible value of index i and index j are:
            0 : PHI0 = "affected by neither D1 or D2"
            1 : PHI1 = "affected by D1 but not D2"
            2 : PHI2 = "affected by D2 but not D1"
            3 : PHI12 = "affected by both D1 and D2"

    patient_counts: 2D num array (float)
        Count of patients at each final age value for each of the 4 possible
        phenotype status.
        First dimension of the array is indexed by the phenotype status:
            0 : PHI0 = "affected by neither D1 or D2"
            1 : PHI1 = "affected by D1 but not D2"
            2 : PHI2 = "affected by D2 but not D1"
            3 : PHI12 = "affected by both D1 and D2"
        Second dimension of the array is indexed by the age_index of
        final_age_array.
            
        Stored value in array as float becuase patient counts might be 
        normalized.

    final_age_array : 1D numpy array (int)
        Array of monotonically increasing (distinct) age values.
        This allows the age_index in patients_countto be convert to actual 
        age_value.

    prval_norm_func : python function object, optional 
        A python function that take as input a 4 element raw count array
        and return back a 4 element normalized count array.

    Returns
    -------
    log_likelihood: float
        The log likelihood of the log likelihood of the genetic pentrance 
        model given the observed EMR data.
    """

    if genetic_penetrance.get_overlap_type() == "independent":
         rho12 = 0.0

    # Compute likelihood using Eq. 51 in Appendix 2 of Rzhetsky et al. 2007.
    log_likelihood = 0.0

    phi_infinity_probs = genetic_penetrance.compute_probs(rho1, rho2, rho12)

    # Normalize the counts
    if prval_norm_func != None:
        # sum over the age axis.
        raw_count = np.sum(patient_counts, axis=1)
        norm_count = prval_norm_func(raw_count)

        norm_factor = np.ones(4)

        for n in xrange(len(norm_count)):
            if raw_count[n] >= 1e-20:
                norm_factor[n] = norm_count[n] / raw_count[n]
            else:
                norm_factor[n] = 0.0

    # TODO: Convert this computation into a matrix formulation to speed up
    # the computation.
    for phi_t_index in [0, 1, 2, 3]:
      
        for age_index, age_val in enumerate(final_age_array):

            patient_count = patient_counts[phi_t_index, age_index]

            # skip bin with no patients
            if patient_count < 1e-20: continue

            if prval_norm_func != None:
               patient_count *= norm_factor[phi_t_index]

            patient_likelihood = 0.0

            for phi_inf_index in [0, 1, 2, 3]:

                age_of_onset_func = joint_age_of_onset_funcs[phi_t_index,
                                                             phi_inf_index]

                age_of_onset_prob = age_of_onset_func(age_val) + 1e-10

                patient_likelihood += (age_of_onset_prob * 
                                       phi_infinity_probs[phi_inf_index])

            if patient_likelihood == 0.0:
                raise ValueError("patient_likelihood is 0.0")


            log_likelihood += patient_count * np.log(patient_likelihood)

    return log_likelihood

def __compute_deriv_log_likelihood(rho1, rho2, rho12,
                                   genetic_penetrance,
                                   joint_age_of_onset_funcs,
                                   patient_counts,
                                   final_age_array,
                                   prval_norm_func = None):
    """Compute the derivative of the log_likelihod with respect to rho1, rho2
    and rho12.

    Returns
    -------
    deriv_log_likelihood: 1D float numpy array (size = 3)
        deriv_log_likelihood[0] = derivative of log_likelihood wrt to rho1
        deriv_log_likelihood[1] = derivative of log_likelihood wrt to rho2
        deriv_log_likelihood[2] = derivative of log_likelihood wrt to rho12
    """

    if genetic_penetrance.get_overlap_type() == "independent":
         rho12 = 0.0

    # 2D numpy array, shape = (4, 3)
    deriv_phi_infinity_probs = genetic_penetrance.compute_deriv_probs(rho1,
                                                                      rho2,
                                                                      rho12)

    # 1D numpy array, size = 4
    phi_infinity_probs = genetic_penetrance.compute_probs(rho1, rho2, rho12)

    # Normalize the counts
    if prval_norm_func != None:
        # sum over the age axis.
        raw_count = np.sum(patient_counts, axis=1)
        norm_count = prval_norm_func(raw_count)

        norm_factor = np.ones(4)

        for n in xrange(len(norm_count)):
            if raw_count[n] >= 1e-20:
                norm_factor[n] = norm_count[n] / raw_count[n]
            else:
                norm_factor[n] = 0.0

    # Derivative of log-likelihood wrt to rho1, rho2, and rho12
    deriv_log_likelihood = np.zeros(3, dtype=np.float)

    for phi_t_index in [0, 1, 2, 3]:
      
        for age_index, age_val in enumerate(final_age_array):

            patient_count = patient_counts[phi_t_index, age_index]

            # skip bin with no patients
            if patient_count < 1e-20: continue

            if prval_norm_func != None:
               patient_count *= norm_factor[phi_t_index]

            patient_likelihood = 0.0
            deriv_patient_likelihood = np.zeros(3, dtype=np.float)

            for phi_inf_index in [0, 1, 2, 3]:

                age_of_onset_func = joint_age_of_onset_funcs[phi_t_index,
                                                             phi_inf_index]

                age_of_onset_prob = age_of_onset_func(age_val) + 1e-10

                patient_likelihood += (age_of_onset_prob *
                                       phi_infinity_probs[phi_inf_index])

                deriv_patient_likelihood += (
                    age_of_onset_prob *
                    deriv_phi_infinity_probs[phi_inf_index])

            deriv_log_likelihood += (patient_count *
                                     (1.0 / patient_likelihood) *
                                     deriv_patient_likelihood)

    return deriv_log_likelihood
