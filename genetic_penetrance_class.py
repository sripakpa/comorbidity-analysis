#!/usr/bin/env python

import math

import numpy as np

import poisson

class GeneticPenetrance(object):
    """GeneticPenetrance class

    This class compute the age-integrated phenotype probability for a pair of
    diseases D1 and D2 using the genetic penetrance models proposed by
    Rzhetsky et al. 2007.

    Concretely, this class computes:
        phi_infty_probs[0] = P(phi(infty) = PHI0; rho1, rho2, rho12)
        phi_infty_probs[1] = P(phi(infty) = PHI1; rho1, rho2, rho12)
        phi_infty_probs[2] = P(phi(infty) = PHI2; rho1, rho2, rho12)
        phi_infty_probs[3] = P(phi(infty) = PHI12; rho1, rho2, rho12)

    For details, see please SI Appendix 2 of Rzhetsky et al. 2007.

    Assume, using the Poission variant of the model for now. Might implement
    the binomail variant of the model in the future.

    Parameters
    ----------
    verbose : bool, optional
        Verbose output.

    tau1: integer
        Model's parameter indicating the minimum number of deleterious mutation
        in S1 or S12, the combined region of the genome that predisposes the
        polymorphisms' bearers to disease D1
        The tau2 values used in Rzhetsky et al. 2007 were tau1 = 1 or tau1 =  3.

    tau2: integer
        Model's parameter indicating the minimum number of deleterious mutation
        in S2 or S12, the combined region of the genome that predisposes the
        polymorphisms' bearers to disease D2
        The tau2 values used in Rzhetsky et al. 2007 were tau2 = 1 or tau2 =  3.

    overlap_type: integer (0, 1 or 2)
        Model's parameter describing the type of genetic overlap used in the
        model:
            0: "cooperation:
                TODO: Describe this!
            1: "competition":
                TODO: Describe this!
            2: "independent":
            TODO: Describe this!

    threshold_type: integer (0 or 1)
        Model's parameter describing the type of genetic penetrance used in 
        the model.
            0: "sharp":
                TODO: Describe this!
            1: "soft":
                TODO: Describe this!

    Attributes
    ----------
    overlap_type_dict: dictionary
        Dictionary mapping overlap_type string to correspond int.
            "cooperation: 0 
            "competition" : 1
            "independent" : 2

    threshold_type_dict: dictionary
            "sharp": 0
            "soft": 0

        Dictionary mapping threshold_type string to correspond int.
    """

    def __init__(self, tau1, tau2, overlap_type, threshold_type,
                 verbose=False):
        """Instantiate variables in GeneticPenetrance class."""

        self.overlap_type_dict = self.__create_overlap_type_dict()
        self.threshold_type_dict = self.__create_threshold_type_dict()

        self.__validate_parameters(tau1, tau2, overlap_type, threshold_type)

        if isinstance(overlap_type, str): 
            overlap_type = self.overlap_type_dict[overlap_type]

        if isinstance(threshold_type, str): 
            threshold_type = self.threshold_type_dict[threshold_type]

        self.tau1 = tau1
        self.tau2 = tau2

        self.overlap_type = overlap_type
        self.threshold_type = threshold_type

        self.verbose = verbose

    # Public methods
    def get_tau1(self):
        """Return self.tau1."""
        return self.tau1

    def get_tau2(self):
        """Return self.tau2."""
        return self.tau2

    def get_overlap_type(self):
        """Return overlap_type string."""

        for string in self.overlap_type_dict:

            if self.overlap_type_dict[string] == self.overlap_type:
                return string

        raise RuntimeError("Cannot find overlap_type string.")

    def get_threshold_type(self):
        """Return threshold_type string."""

        for string in self.threshold_type_dict:

            if self.threshold_type_dict[string] == self.threshold_type:
                return string

        raise RuntimeError("Cannot find threshold_type string.")

    def compute_probs(self, rho1, rho2, rho12):
        """Compute and return probability of phi(infty), the age-integrated
        phenotypes.

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
            number of deleterious polymorphisms in S12, the region of the
            genome that predisposes the polymorphisms' bearers to both disease
            D1 and disease D2
            This parameter is set to zero in the independence model.

        Returns:
        --------
        phi_infinity_probs : 1D float numpy array (size = 4)
            The age-integrated phenotype for all 4 possible phenotype 2 
            diseases phenotype status:
                phi_infty_probs[0] = P(phi(infty) = PHI0; rho1, rho2, rho12)
                phi_infty_probs[1] = P(phi(infty) = PHI1; rho1, rho2, rho12)
                phi_infty_probs[2] = P(phi(infty) = PHI2; rho1, rho2, rho12)
                phi_infty_probs[3] = P(phi(infty) = PHI12; rho1, rho2, rho12)
        """

        if not isinstance(rho1, float):
            raise ValueError("rho1 must be a float")

        if not isinstance(rho2, float):
            raise ValueError("rho2 must be a float")

        if not isinstance(rho12, float):
            raise ValueError("rho12 must be a float")

        if self.overlap_type == 2:  # independent model
            if math.fabs(rho12) > 1e-20:
                raise AssertionError("rho12 must be 0.0 in independent model.")

        if self.threshold_type == 0:  # sharp penetrance model
            return self.__probs_sharp_threshold_wrap(rho1, rho2, rho12)
        else:
            return self.__probs_soft_threshold(rho1, rho2, rho12)

    def compute_deriv_probs(self, rho1, rho2, rho12):
        """Compute and return derivative of phi(infty) with respect to rho1,
        rho2, and rho12.

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
            number of deleterious polymorphisms in S12, the region of the 
            genomethat predisposes the polymorphisms' bearers to both disease
            D1 and disease D2
            This parameter is set to zero in the independence model.

        Returns:
        --------
        deriv_probs: 2D float numpy array, shape = (4, 3)
            derivative of phi(infty) with respect to rho1, rho2 and rho12.

                                P(phi(infty) = PHI[i]; rho1, rho2, rho12)
            deriv_probs[i,j] =  -----------------------------------------
                                              rho[j]

            ith index:
                PHI[0] = PHI0, PHI[1] = PHI1, PHI[2] = PHI2 and PHI[3] = PHI12

            jth index:
                rho[0] = rho1, rho[1] = rho2 and rho[2] = rho12
        """

        if not isinstance(rho1, float):
            raise ValueError("rho1 must be a float")

        if not isinstance(rho2, float):
            raise ValueError("rho2 must be a float")

        if not isinstance(rho12, float):
            raise ValueError("rho12 must be a float")

        if self.overlap_type == 2:  # independent model
            if math.fabs(rho12) > 1e-20:
                raise AssertionError("rho12 must be 0.0 in independent model.")

        if self.threshold_type == 0:  # sharp penetrance model
            return self.__deriv_probs_sharp_threshold(rho1, rho2, rho12)
        else:
            return self.__deriv_probs_soft_threshold(rho1, rho2, rho12)


    # Private methods
    def __probs_sharp_threshold_wrap(self, rho1, rho2, rho12):
        """Deal with negative rhos."""

        rho1_mock = rho1
        rho2_mock = rho2
        rho12_mock = rho12

        if rho1 < 0.0: rho1_mock = 0.0
        if rho2 < 0.0: rho2_mock = 0.0
        if rho12 < 0.0: rho2_mock = 0.0

        phi_infty_probs = self.__probs_sharp_threshold(rho1_mock,
                                                       rho2_mock,
                                                       rho12_mock)

        return phi_infty_probs

    def __probs_sharp_threshold(self, rho1, rho2, rho12):
        """Compute and return probability of phi(infty), the age-integrated
        phenotypes using the sharp threshold model.

        Please refer to section 6.2 in the SI Appendix 2 of Rzhetsky et al.
        2007 for details.

        Returns
        -------
        phi_infty_probs : 1D float numpy array (size = 4)

        Notes
        -----
        poisson.pmf(k, mu) is probability mass function of the Poisson
        distribution with parameter mu evualated at k.

        poisson.cdf(kmax, mu) is cumulative of Poisson probability mass
        function from k = 0, 1, 2 to kmax.
        """

        prob_noD1 = 0.0
        prob_noD2 = 0.0
        prob_noD1_and_noD2 = 0.0

        tau1 = self.tau1
        tau2 = self.tau2
        tau12 = max(tau1, tau2)

        if self.overlap_type == 0:  # coorperative model

            indices = [(k1, k2, k12) for k1 in xrange(tau1)
                                     for k2 in xrange(tau2)
                                     for k12 in xrange(tau12)]

            prob_noD1 = poisson.cdf(tau1 - 1, rho1 + rho12)
            prob_noD2 = poisson.cdf(tau2 - 1, rho2 + rho12)

            for (k1, k2, k12) in indices:
                no_penetrance = k1 + k12 < tau1 and k2 + k12 < tau2

                if no_penetrance:
                    prob_noD1_and_noD2 += (poisson.pmf(k1, rho1) *
                                           poisson.pmf(k2, rho2) *
                                           poisson.pmf(k12, rho12))

        elif self.overlap_type == 1 and rho12 > 0.0:  # competition model
            raise NotImplementedError("competition model not implemented.")

        else:  # independent model, rho12 does not contribute!

            prob_noD1 = poisson.cdf(tau1 - 1, rho1)
            prob_noD2 = poisson.cdf(tau2 - 1, rho2)

            indices = [(k1, k2) for k1 in xrange(tau1)
                                for k2 in xrange(tau2)]

            for (k1, k2) in indices:
                no_penetrance = k1 < tau1 and k2 < tau2

                if no_penetrance:
                    prob_noD1_and_noD2 += (poisson.pmf(k1, rho1) *
                                           poisson.pmf(k2, rho2))

         # Instantiate phi_infinity_probs
        phi_infty_probs = np.zeros(4, dtype=np.float)

        # prob(noD1 and noD2)
        phi_infty_probs[0] = prob_noD1_and_noD2

        # prob(yesD1 and noD2)
        phi_infty_probs[1] = prob_noD2 - prob_noD1_and_noD2

        # prob(noD1 and yesD2)
        phi_infty_probs[2] = prob_noD1 - prob_noD1_and_noD2

        # prob(yesD1 and yesD2)
        # use the fact that probability must add up to 1.
        phi_infty_probs[3] = 1.0 - np.sum(phi_infty_probs[0:3])

        return phi_infty_probs

    def __deriv_probs_sharp_threshold(self, rho1, rho2, rho12):
        """Compute and return derivative of probability of phi(infty) with
        respect to rho1, rho2 and rho12 using the sharp threshold model.

        Returns
        -------
        dprobs_drhos : 2D float numpy array, shape = (4, 3)
        """

        # 3 elements in array are derivative wrt to rho1, rho2 and rho12.
        deriv_prob_noD1 = np.zeros(3, dtype=np.float)
        deriv_prob_noD2 = np.zeros(3, dtype=np.float)
        deriv_prob_noD1_and_noD2 = np.zeros(3, dtype=np.float)

        tau1 = self.tau1
        tau2 = self.tau2
        tau12 = max(tau1, tau2)

        if self.overlap_type == 0:  # coorperative model

            indices = [(k1, k2, k12) for k1 in xrange(tau1)
                                     for k2 in xrange(tau2)
                                     for k12 in xrange(tau12)]

            deriv_prob_noD1[0] = poisson.dcdf_dmu(tau1 - 1, rho1 + rho12)
            deriv_prob_noD1[1] = 0.0
            deriv_prob_noD1[2] = poisson.dcdf_dmu(tau1 - 1, rho1 + rho12)

            deriv_prob_noD2[0] = 0.0
            deriv_prob_noD2[1] = poisson.dcdf_dmu(tau2 - 1, rho2 + rho12)
            deriv_prob_noD2[2] = poisson.dcdf_dmu(tau2 - 1, rho2 + rho12)

            for (k1, k2, k12) in indices:

                no_penetrance = k1 + k12 < tau1 and k2 + k12 < tau2

                if no_penetrance:

                    deriv_prob_noD1_and_noD2[0] = (
                        poisson.dpmf_dmu(k1, rho1) *
                        poisson.pmf(k2, rho2) *
                        poisson.pmf(k12, rho12))

                    deriv_prob_noD1_and_noD2[1] = (
                        poisson.pmf(k1, rho1) *
                        poisson.dpmf_dmu(k2, rho2) *
                        poisson.pmf(k12, rho12))

                    deriv_prob_noD1_and_noD2[2] = (
                        poisson.pmf(k1, rho1)*
                        poisson.pmf(k2, rho2) *
                        poisson.dpmf_dmu(k12, rho12))

        elif self.overlap_type == 1 and rho12 > 0.0:  # competition model
            raise NotImplementedError("competition model not implemented.")

        else:  # independent model, rho12 does not contribute!

            deriv_prob_noD1[0] = poisson.dcdf_dmu(tau1 - 1, rho1)
            deriv_prob_noD1[1] = 0.0
            deriv_prob_noD1[2] = 0.0

            deriv_prob_noD2[0] = 0.0
            deriv_prob_noD2[1] = poisson.dcdf_dmu(tau2 - 1, rho2)
            deriv_prob_noD2[2] = 0.0

            indices = [(k1, k2) for k1 in xrange(tau1)
                                for k2 in xrange(tau2)]

            for (k1, k2) in indices:
                no_penetrance = k1 < tau1 and k2 < tau2

                if no_penetrance:

                    deriv_prob_noD1_and_noD2[0] = (
                        poisson.dpmf_dmu(k1, rho1) *
                        poisson.pmf(k2, rho2))

                    deriv_prob_noD1_and_noD2[1] = (
                        poisson.pmf(k1, rho1) *
                        poisson.dpmf_dmu(k2, rho2))

                    deriv_prob_noD1_and_noD2[2] = 0.0

        # Derivative of phi_infty_probs wrt to rho1, rho2, and rho12
        deriv_probs = np.zeros([4, 3], dtype=np.float)

        # deriv of prob(noD1 and noD2)
        deriv_probs[0] = deriv_prob_noD1_and_noD2

        # deriv of prob(yesD1 and noD2)
        deriv_probs[1] = deriv_prob_noD2 - deriv_prob_noD1_and_noD2

        # deriv of prob(noD1 and yesD2)
        deriv_probs[2] = deriv_prob_noD1 - deriv_prob_noD1_and_noD2

        # deriv of prob(yesD1 and yesD2)
        deriv_probs[3] = -np.sum(deriv_probs[0:3])

        return deriv_probs

    def __probs_soft_threshold(self, rho1, rho2 , rho12):
        """Compute and return probability of phi(infty), the 
        age-integrated phenotypes using the soft_threshold model."""

        raise NotImplementedError("soft threshold model not implemented.")

    def __deriv_probs_soft_threshold(self, rho1, rho2 , rho12):
        """Compute and return derivative of probability of phi(infty)
        with respect to rho1, rho2, and rho12 using the soft_threshold model.
        """

        raise NotImplementedError("soft threshold model not implemented.")

    def __create_overlap_type_dict(self):
        """Create the overlap_type dictionary object."""

        overlap_type_dict = {"cooperation" : 0,
                             "competition" : 1,
                             "independent" : 2}

        return overlap_type_dict

    def __create_threshold_type_dict(self):
        """Create the threshold_type dictionary object."""

        threshold_type_dict = {"sharp" : 0, "soft" : 1}

        return threshold_type_dict

    def __validate_parameters(self, tau1, tau2, overlap_type, threshold_type):
        """Validate the parameters inputted into the class."""

        if not isinstance(tau1, int):
            raise ValueError("tau1 must be a int")

        if tau1 <= 0:
            raise ValueError("tau1 must be a positive, non-zero integer.")

        if not isinstance(tau2, int):
            raise ValueError("tau2 must be a int")

        if tau2 <= 0:
            raise ValueError("tau2 must be a positive, non-zero integer.")

        if isinstance(overlap_type, str):
            if overlap_type not in self.overlap_type_dict:
                raise ValueError("invalid overlap_type %s" % overlap_type)
        else:
            if not isinstance(overlap_type, int):
                raise TypeError("overlap_type need to be str or int type.")

        if isinstance(threshold_type, str):
            if threshold_type not in self.threshold_type_dict:
                raise ValueError("invalid threshold_type %s" % threshold_type)
        else:
            if not isinstance(threshold_type, int):
                raise TypeError("threshold_type need to be str or int type.")

