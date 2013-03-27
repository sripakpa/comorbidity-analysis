#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d

class JointAgeOfOnset(object):
    """JointAgeOfOnset class

    This class uses EMR data to empirically construct the conditional 
    probabilities of the age-t phenotype [phi(t)] given the ultimate
    (age-integrated) phenotype [phi(infty)] for a pair of diseases D1 and D2.

    Details
    -------
    For each pair of diseases D1 and D2, there are four possible phenotype
    status = {PHI0, PHI1, PHI2, PHI12} corresponding to
        PHI0 = "affected by neither D1 or D2"
        PHI1 = "affected by D1 but not D2"
        PHI2 = "affected by D2 but not D1"
        PHI12 = "affected by both D1 and D2"

    Both phi(t) and phi(infty) can be any of the four phenotype status, so for
    each current age T = t, there are 16 conditional probability to compute
    in total.

    P(phi(t) | phi(infty)) table:

                                  phi(infty)
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

    A1(t), the age-of-onset probability of D1, is the cumulative probability
    that a patient first manifest the disease D1 at or before age T = t 
    given that the patient does eventually manifest the disease in his/her
    life time (T != None).

    A1(t) can be empirically constructed from the EMR data as follow:

            # of patients that first manifest disease D1 before age t
    A1(t) = ----------------------------------------------------------
              # of patients that manifest disease D1 at any age

    Only patients that eventually does manifest the disease D1 contribute to
    the count in either the numerator or the denominator.

    The age that a patient first manifest a disease (age-of-onset) is
    estimated using the age that a ICD-9 associated with the disease first
    appear in the patient's EMR record.

    A2 is defined in a similar manner.

    1 - A1(t) is equivalent to F1(t) in Rzhetsky et al. 2007.

    Parameters
    ----------
    D1orD2_patients : two layer dictionary
        EMR data of patients with have disease D1 or disease D2 or both.
        The patient's ID serves as the dictionary key and each
        D1orD2_patients[ID] represent a patient with the following key-value
        pairs:
            'age': patient's age in years
            'ethnicity': patient's ethnicity (not used by this class)
            'gender': patient's gender (not used by this class)
            'age_of_onsets': a two element list
                age-of-onsets[0] = age-of-onset of disease D1
                age-of-onsets[1] = age-of-onset of disease D2 
                If the patient never have the disease, then the
                corresponding age-of-onset is set to None.

    verbose : bool, optional
        Verbose output.

    Attributes
    ----------
    min_age : integer
        Absolute lower bound on the possible age of a patient.

    max_age : integer
        Absolute upper bound on the possible age of a patient.

    common_age_array : 1D numpy array (int)
        Array of monotonically increasing (distinct) age values. Compute
        each of the 16 conditional probabilities along points in this array.
        Age value at the array's first index is min_age.
        Age value of the array's last index is max_age.

    interp1d_kind : ('linear','nearest','zero','slinear','quadratic,'cubic')
        The kind of interpolation used to interpolate P(phi(t) | phi(infty))
        conditional probabilities. The default kind ('zero') leads to a
        step function.

    joint_age_of_onset_funcs : 4 x 4 numpy array of scipy interp1d objects.
        Each joint_age_of_onsets[i,j] is a scipy 1-D interpolated function
        object representing the P(phi(t) | phi(infty)) conditional
        probabilities.
        Index i is the index of the phi(t) phenotype.
        Index j is the index of the phi(infty) phenotype.
        Possible value of index i and index j are:
            0 : PHI0 = "affected by neither D1 or D2"
            1 : PHI1 = "affected by D1 but not D2"
            2 : PHI2 = "affected by D2 but not D1"
            3 : PHI12 = "affected by both D1 and D2"
    """

    def __init__(self, D1orD2_patients, verbose=False):
        """Instantiate variables in JointAgeOfOnset class."""

        self.verbose = verbose

        self.min_age = 0
        self.max_age = 200  # unlikely that anyone will live to 200 years old!
        self.common_age_array = self.__create_common_age_array(D1orD2_patients)

        self.interp1d_kind = 'zero'  # this leads to a step function.
        self.joint_age_of_onset_funcs = None

        self.__validate_emr_data(D1orD2_patients)
        self.__compute_joint_age_of_onset_funcs(D1orD2_patients)

    # Public methods
    def get_funcs(self):
        """Return joint_age_of_onset_funcs object."""

        return self.joint_age_of_onset_funcs

    # Private methods
    def __compute_joint_age_of_onset_funcs(self, D1orD2_patients):
        """Compute the 4 x 4 joint age-of-onsets conditional probabilities
           using the input EMR dataset.
        """

        A1_array = self.__compute_cumulative_age_of_onset(D1orD2_patients,
                                                          disease_type = 0)

        A2_array = self.__compute_cumulative_age_of_onset(D1orD2_patients,
                                                          disease_type = 1)

        # 4 x 4 numpy array of object type
        joint_age_of_onset_array = np.empty([4,4],dtype=object)

        length = self.common_age_array.size
        # 1st index is phi(t) phenotype, 2nd index is phi(infty) phenotype.
        joint_age_of_onset_array[0,0] = np.ones(length)
        joint_age_of_onset_array[1,0] = np.zeros(length)
        joint_age_of_onset_array[2,0] = np.zeros(length)
        joint_age_of_onset_array[3,0] = np.zeros(length)

        joint_age_of_onset_array[0,1] = 1.0 - A1_array
        joint_age_of_onset_array[1,1] = A1_array
        joint_age_of_onset_array[2,1] = np.zeros(length)
        joint_age_of_onset_array[3,1] = np.zeros(length)

        joint_age_of_onset_array[0,2] = 1.0 - A2_array
        joint_age_of_onset_array[1,2] = np.zeros(length)
        joint_age_of_onset_array[2,2] = A2_array
        joint_age_of_onset_array[3,2] = np.zeros(length)

        joint_age_of_onset_array[0,3] = (1.0 - A1_array) * (1.0 - A2_array)
        joint_age_of_onset_array[1,3] = A1_array * (1.0 - A2_array)
        joint_age_of_onset_array[2,3] = (1.0 - A1_array) * A2_array
        joint_age_of_onset_array[3,3] = A1_array * A2_array

        # Interpolate the arrays.
        self.joint_age_of_onset_funcs = np.empty([4,4],dtype=object) 

        indices = [(i,j) for i in xrange(4) for j in xrange(4)]

        for (i,j) in indices:

            array_ij = joint_age_of_onset_array[i,j]

            func_ij = interp1d(self.common_age_array, array_ij,
                               self.interp1d_kind)

            self.joint_age_of_onset_funcs[i,j] = func_ij

    def __validate_emr_data(self, D1orD2_patients):
        """Validate that the emr_data have all the necessary data and that
           the data are appropriately formatted."""

        for ID in D1orD2_patients:

            patient = D1orD2_patients[ID]

            if 'age_of_onsets' not in patient:
                raise KeyError("Missing age_of_onsets key in patient.")

            age_of_onsets = patient['age_of_onsets']

            if len(age_of_onsets) != 2:
                raise ValueError("age_of_onsets should contain two elements.")

            if age_of_onsets[0] != None:
                if not isinstance(age_of_onsets[0], int):
                    raise TypeError("age_of_onsets[0] has invalid type.")

            if age_of_onsets[1] != None:
                if not isinstance(age_of_onsets[1], int):
                    raise TypeError("age_of_onsets[1] has invalid type.")

    def __create_common_age_array(self, D1orD2_patients):
        """Create an 1D array of distinct age-of-onset values found in the EMR
           data (partition by year) ordered lowest to highest."""

        age_list = []

        for ID in D1orD2_patients:

            patient = D1orD2_patients[ID]

            age_of_onsets = patient['age_of_onsets']

            if age_of_onsets[0] != None and age_of_onsets[0] not in age_list:
                age_list.append(age_of_onsets[0])

            if age_of_onsets[1] != None and age_of_onsets[1] not in age_list:
                age_list.append(age_of_onsets[1])

        if self.min_age not in age_list:
            age_list.append(self.min_age)

        if self.max_age not in age_list:
            age_list.append(self.max_age)

        # Sort the list in place (increasing order).
        age_list.sort()

        if age_list[0] != self.min_age:
            raise ValueError("age_list[0] %d must be min_age." % age_list[0])

        if age_list[-1] != self.max_age:
            raise ValueError("age_list[-1] %d must be max_age." % age_list[-1])

        # Convert to numpy array.
        age_array = np.array(age_list)

        return age_array

    def __create_age2common_index(self):
        """Create a dictionary mapping age value (int) back to its 
        corresponding index in common_age_array."""

        age2index = dict()

        for index, age in enumerate(self.common_age_array):

            if age in age2index:
                raise ValueError("duplicated age value.")

            age2index[age] = index

        return age2index

    def __compute_cumulative_age_of_onset(self, D1orD2_patients,
                                          disease_type):
        """Compute an array of empirical age-of-onset probability at each age
        point in the common_age_array.

        The length of A_array should be the same as the length of the 
        common_age_array.

                # of patients that first manifest disease D1 before age t
        A1(t) = ----------------------------------------------------------
                  # of patients that manifest disease D1 at any age
        """

        if disease_type not in [0,1]:
            raise ValueError("disease_type must be 0 or 1.")

        age2common_age_index = self.__create_age2common_index()

        # Instantiate the age_of_onset_counts array
        length = self.common_age_array.size
        age_of_onset_counts = np.zeros(length, dtype=np.float)

        # Tally age_of_onset_counts at each point along the common_age_array.
        for ID in D1orD2_patients:
            age_of_onset = D1orD2_patients[ID]['age_of_onsets'][disease_type]

            if age_of_onset == None: continue

            if not isinstance(age_of_onset, int):
                raise TypeError("age_of_onset must be an int.")

            common_age_index = age2common_age_index[age_of_onset]

            age_of_onset_counts[common_age_index] += 1.0

        total_counts = np.sum(age_of_onset_counts)

        # Make sure total_counts is not zero. This can happen in cases where
        # there is zero patient of specific sex with disease. For example, male
        # patient with female breast cancer.
        if total_counts == 0: total_counts = 1

        # Convert to a cumulative count.
        age_of_onset_cum_counts = np.cumsum(age_of_onset_counts)

        # Convert to probability
        age_of_onset_cum_prob = age_of_onset_cum_counts / total_counts

        return age_of_onset_cum_prob
