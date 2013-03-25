#!/usr/bin/env python

import numpy as np

class JointFinalAge(object):
    """JointAgeOfOnset class

    This class compute the count of patients as a distribution of their final
    age value. The patients are seperated by their 2 disease phenotype status
    and the count are computed seperately for each phenotype status.

    Concretely, this class:

    (1) create an ordered array of distinct final age values
    found in the EMR dataset.

    (2) compute the count of patients at each final age value seperately for
    each phenotyp status  on stored it as 2D array with dimensions 
    (num_phenotype_status x num_distinct_ages)

    Details
    -------
    For each pair of diseases D1 and D2, there are four possible phenotype
    status = {PHI0, PHI1, PHI2, PHI12} corresponding to
        PHI0 = "affected by neither D1 or D2"
        PHI1 = "affected by D1 but not D2"
        PHI2 = "affected by D2 but not D1"
        PHI12 = "affected by both D1 and D2"

    Final age is the highest age value that was recorded in the
    EMR dataset for the patient.

    Parameters
    ----------
    D1orD2_patients : two layer dictionary
        EMR data of patients with have disease D1 or disease D2 or both.
        The patient's ID serves as the dictionary key and each EMR_data[ID]
        contains the following key-value pairs.
            'age': patient's age in years
            'ethnicity': patient's ethnicity (not used by this class)
            'gender': patient's gender (not used by this class)
            'age_of_onsets': a two element list
                age_of_onsets[0] = age-of-onset of disease D1
                age_of_onsets[1] = age-of-onset of disease D2 
                If the patient never have the disease, that the
                corresponding age-of-onset is set to None

    noD1D2_patients : two layer dictionary
        EMR data of patients without the disease D1 and without disease D2.
        Same data structure as D1orD2_patients.

    verbose : bool, optional
        Verbose output.

    Attributes
    ----------
    final_age_array : 1D numpy array (int)
        Array of monotonically increasing (distinct) age values.

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
    """

    def __init__(self, D1orD2_patients, noD1D2_patients, verbose=False):
        """Instantiate variables in JointFinalAge class."""

        self.verbose = verbose

        self.__validate_emr_data(D1orD2_patients)
        self.__validate_emr_data(noD1D2_patients)

        self.final_age_array = self.__create_final_age_array(D1orD2_patients,
                                                             noD1D2_patients)

        self.patient_counts = self.__create_patient_counts(D1orD2_patients,
                                                           noD1D2_patients)

    # Public methods
    def get_final_age_array(self):
        """Return the final_age_array object."""

        return self.final_age_array

    def get_patient_counts(self):
        """Return the patient_counts object."""

        return self.patient_counts

    # Private methods
    def __validate_emr_data(self, EMR_data):
        """Validate that the age field in the emr_data are correctly
           formatted."""

        for ID in EMR_data:

            patient = EMR_data[ID]

            if 'age' not in patient:
                raise KeyError("Missing age key in EMR_data[ID].")

            final_age = patient['age']

            if not isinstance(final_age, int):
                raise TypeError("final_age must be an int.")


    def __create_final_age_array(self, D1orD2_patients, noD1D2_patients):
        """Create an 1D array of distinct final age values found in the EMR
           data (partition by year) ordered lowest to highest."""

        age_list = []

        for ID in D1orD2_patients:
            age = D1orD2_patients[ID]['age']

            if age not in age_list:
                age_list.append(age)

        for ID in noD1D2_patients:
            age = noD1D2_patients[ID]['age']

            if age not in age_list:
                age_list.append(age)

        # Sort the list in place (increasing order).
        age_list.sort()

        # Convert to numpy array.
        age_array = np.array(age_list)

        return age_array

    def __create_age2index(self):
        """Create a dictionary mapping final age (int) back to its 
        corresponding index in final_age_array."""

        age2index = dict()

        for index, age in enumerate(self.final_age_array):

            if age in age2index:
                raise ValueError("duplicated age value.")

            age2index[age] = index

        return age2index

    def __create_patient_counts(self, D1orD2_patients, noD1D2_patients):
        """compute the count of patients at each final age value seperately for
           each phenotyp status."""

        # Instantiate the patient_counts 2D array
        num_phenotypes = 4
        num_distinct_ages = self.final_age_array.size
        dimensions = (num_phenotypes, num_distinct_ages)
        patient_counts = np.zeros(dimensions, dtype=np.float64)

        age2index = self.__create_age2index()

        # Tally the counts for the "with diseases" patients
        for ID in D1orD2_patients:

            patient = D1orD2_patients[ID]

            has_D1 = patient['age_of_onsets'][0] != None
            has_D2 = patient['age_of_onsets'][1] != None

            if has_D1 and has_D2:
                phenotype = 3
            elif has_D2:
                phenotype = 2
            elif has_D1:
                phenotype = 1
            else:
                raise ValueError("patient should have D1, D2 or both.")

            age = patient['age']

            if age not in age2index:
                raise KeyError("age %d not found in age2index." % age)

            patient_counts[phenotype, age2index[age]] += 1.0

        # Tally the counts for the "without diseases" patients
        for ID in noD1D2_patients:

            patient = noD1D2_patients[ID]

            # 0 : PHI0, "affected by neither D1 or D2" 
            phenotype = 0 
            age = patient['age']

            if age not in age2index:
                raise KeyError("age %d not found in age2index." % age)

            patient_counts[phenotype, age2index[age]] += 1.0

        return patient_counts

