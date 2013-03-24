#!/usr/bin/env python
"""Compute the empirical age-of-onset function from a 1 disease EMR dataset.
"""

import numpy as np
from scipy.interpolate import interp1d

def __validate_emr_data(D1_patients):
    """Validate that the emr_data have all the necessary data and that
       the data are appropriately formatted."""

    for ID in D1_patients:

        patient = D1_patients[ID]
        age_of_onsets = patient['age_of_onsets']

        # EMR data should store age-of-onset data for only 1 disease
        if len(age_of_onsets) != 1:
            raise ValueError("age_of_onsets should contain one elements.")

        if not isinstance(age_of_onsets[0], int):
            raise TypeError("age_of_onset must be an int.")

def __create_age_array(D1_patients,  min_age, max_age):
    """Create an 1D array of distinct age-of-onset values found in the EMR
       data (partition by year) ordered lowest to highest."""

    # Create an sorted list of distinct age values.
    age_list = []

    for ID in D1_patients:

        patient = D1_patients[ID]
        age_of_onsets = patient['age_of_onsets']

        age_of_onset = age_of_onsets[0]

        if not isinstance(age_of_onset, int):
            raise TypeError("age_of_onset must be an int.")

        if age_of_onset not in age_list:
            age_list.append(age_of_onset)

    if min_age not in age_list: 
        age_list.append(min_age)
    if max_age not in age_list: 
        age_list.append(max_age)

    age_list.sort()  # Sort the list in place (increasing order).

    if age_list[0] != min_age:
        raise ValueError("age_list[0] %d must be min_age" % age_list[0])

    if age_list[-1] != max_age:
        raise ValueError("age_list[-1] %d must be min_age" % age_list[-1])

    # Convert to numpy array
    age_array = np.array(age_list)

    return age_array

def __create_age2index(age_array):
    """Create a dictionary mapping age value (int) back to its 
    corresponding index in age_array."""

    age2index = dict()

    for index, age in enumerate(age_array):

        if age in age2index: raise ValueError("duplicated age value.")

        age2index[age] = index

    return age2index 

def __tally_age_of_onset(D1_patients, age_array):
    """Tally age_of_onset counts at each point along the age_array and return
       counts as an 1D numpy float array."""

    # Instantiate the age_of_onset_counts array
    length = age_array.size
    age_of_onset_counts = np.zeros(length, dtype=np.float64)

    age2index = __create_age2index(age_array)

    for ID in D1_patients:

        patient = D1_patients[ID]
        age_of_onset = patient['age_of_onsets'][0]

        if not isinstance(age_of_onset, int):
            raise TypeError("age_of_onset must be an int.")

        age_index = age2index[age_of_onset]

        age_of_onset_counts[age_index] += 1.0

    return age_of_onset_counts

def create_age_of_onset_func(D1_patients, kind='zero'):
    """Generate the age-of-onset function from the provided EMR data.

    The age-of-onset function give the cumulative probability that a 
    disease/disease manifest itself before or at any given age, conditioned
    on the outcome that the patient does eventually get the disease sometime
    in his or her lifespan.

    Parameters
    ----------
    D1_patients: two layer dictionary
        EMR data of patients with have some disease D1
        The patient's ID serves as the dictionary key and each EMR_data[ID]
                contains the following key-value pairs.
                    'age': patient's age in years
                    'ethnicity': patient's ethnicity (not used by this class)
                    'gender': patient's gender (not used by this class)
                    'age_of_onsets': a one (int) element list
                        age_of_onsets[0] = age-of-onset of disease D1

    kind: ('linear','nearest', 'zero','slinear','quadratic,'cubic'), optional
        Specifies the type of function interpolation.

    Returns
    -------
    age_of_onset_func: scipy interp1d objects
        A scipy 1-D interpolated function object representing the age-of-onset
        cummulative probability for for the 
    """

    __validate_emr_data(D1_patients)

    min_age = 0
    max_age = 200  # unlikely that anyone will live to 200 years old!

    age_array = __create_age_array(D1_patients, min_age, max_age)

    age_of_onset_counts = __tally_age_of_onset(D1_patients, age_array)

    total_counts = np.sum(age_of_onset_counts)

    # Convert to a cumulative count.
    age_of_onset_cum_counts = np.cumsum(age_of_onset_counts)

    # Convert to probability
    age_of_onset_cum_prob = age_of_onset_cum_counts / total_counts

    func = interp1d(age_array, age_of_onset_cum_prob, kind=kind)

    return func, age_array, age_of_onset_cum_prob
