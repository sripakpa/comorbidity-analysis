#!/usr/bin/env python
"""Create a function to normalize EMR data so that the raw EMR disease 
prevalence to match the general population disease prevalence."""

import numpy as np

def create_prval_norm_func(D1, D2,
                           disease2emr_count,
                           total_patient_count,
                           prevalence_file,
                           norm_prval_method,
                           verbose = False):
    """Create a function to normalize the raw EMR disease prevalence to match
    the general population disease prevalence

    EMR and general population disease pravalance can differ due to 
    (1) Under-representation or over-representation of healthy patients.
    (2) Differing hospital visit rate to different diseases.

    Parameters
    ----------
    disease2emr_count: dictionary (string to int)
        Dictionary mapping disease name to count of EMR patients with the 
        disoder

    total_patient_count: integer
        Total count of unique patients in the EMR dataset

    prevalence_file: string
        Location of the prevalence data file

    norm_prval_method : [None, "rzhetsky", "max", "min",
                         "avg", "wts_avg", "sum"]
        Protocol to normalize the prevalence data.

    verbose: bool, optional
        Verbose output.

    Returns
    -------
    prval_norm_func: python function object
        A python function that take as input a 4 element raw count array
        and return back a 4 element normalized count array.

    """

    population_prval = __import_prevalence_file(prevalence_file)

    norm_factors = __compute_prval_norm_factors(D1, D2,
                                                disease2emr_count,
                                                total_patient_count,
                                                population_prval,
                                                norm_prval_method)

    if verbose:
        print "----------create_prval_norm_func----------"
        print "norm_prval_method = %s" % norm_prval_method
        print "D1_emr_count = %d "% disease2emr_count[D1]
        print "D2_emr_count = %d " % disease2emr_count[D2]
        print "total_patient_count = %d " % total_patient_count
        print "D1_pop_prval = %.3f" % population_prval[D1]
        print "D2_pop_prval = %.3f" % population_prval[D2]
        print "norm_factors[PHI1] = %.3f" % norm_factors["PHI1"]
        print "norm_factors[PHI2] = %.3f" % norm_factors["PHI2"]
        print "norm_factors[PHI12] = %.3f" % norm_factors["PHI12"]
        print "------------------------------------------"

    def prval_norm_func(raw_counts):
        """Normalize the raw (EMR) counts so that the raw (EMR) prevalence
        matches the general population disease prevalence.

        Raw_counts can be actual count of EMR patients or perhaps fractional
        counts (probabilities).

        Parameters
        ----------
        raw_counts: 1D float numpy array (size = 4)
            raw_counts[0] = Raw PHI0 data, raw_counts[1] = Raw PHI1 data
            raw_counts[2] = Raw PHI2 data, raw_counts[3] = Raw PHI12 data

        Returns
        -------
        norm_counts: 1D float numpy array (size = 4)
            normalized version of raw_counts.
        """

        # check that raw_counts is indeed a 1D float numpy array of size 4.
        if not isinstance(raw_counts, np.ndarray):
            raise TypeError("raw_counts must be a numpy array.")

        if raw_counts.dtype != np.float:
            raise TypeError("raw_counts.dtype should be float")

        if len(raw_counts.shape) != 1 or raw_counts.size != 4:
            raise TypeError("raw_counts should be a 4 element 1D numpy array")

        total_counts = np.sum(raw_counts)
        norm_counts = np.zeros(4, dtype=np.float)

        norm_counts[1] = norm_factors["PHI1"] * raw_counts[1]
        norm_counts[2] = norm_factors["PHI2"] * raw_counts[2]
        norm_counts[3] = norm_factors["PHI12"] * raw_counts[3]

        # We normalize so that the total counts (probabilties) is conserved.
        norm_counts[0] = total_counts - np.sum(norm_counts[1:4])

        if norm_counts[0] < 0.0:
            raise RuntimeError("encouter norm_counts[0] <= 0.0.")

        return norm_counts

    return prval_norm_func

def __compute_prval_norm_factors(D1, D2,
                                 disease2emr_count,
                                 total_patient_count,
                                 population_prval,
                                 norm_prval_method):
    """Compute return the factors to normalize the raw EMR prevalence to match
    the general population prevalence"""

    norm_factors = dict()

    if norm_prval_method == None:

        norm_factors["PHI1"] = 1.0
        norm_factors["PHI2"] = 1.0
        norm_factors["PHI12"] = 1.0

    elif norm_prval_method == "Rzhetsky":

        rzhetsky_prval_ratio = (
            __compute_rzhetsky_prval_ratio(D1, D2,
                                           disease2emr_count,
                                           total_patient_count,
                                           population_prval,
                                           norm_prval_method))

        norm_factors["PHI1"] = 1.0 / rzhetsky_prval_ratio
        norm_factors["PHI2"] = 1.0 / rzhetsky_prval_ratio
        norm_factors["PHI12"] = 1.0 / rzhetsky_prval_ratio

    else:
        """
        For max, min, avg, wts_avg, sum and independent scheme:
                               P(V | D1notD2)    EMR Prevalence of D1
            prval_ratio_PHI1 = -------------- =~ --------------------
                               P(V | D1notD2)    Pop Prevalence of D1

                               P(V | D2notD1)    EMR Prevalence of D2
            prval_ratio_PHI2 = -------------- =~ --------------------
                                    P(V)         Pop Prevalence of D2

                                P(V | D1andD2)
            prval_ratio_PHI12 = --------------, varies by norm_prval_method.
                                    P(V)    
        """

        emr_prval_D1 = float(disease2emr_count[D1]) / total_patient_count
        emr_prval_D2 = float(disease2emr_count[D2]) / total_patient_count

        pop_prval_D1 = population_prval[D1]
        pop_prval_D2 = population_prval[D2]

        prval_ratio_PHI1 = emr_prval_D1 / pop_prval_D1 
        prval_ratio_PHI2 = emr_prval_D2 / pop_prval_D2 

        prval_ratio_PHI12 = __get_PHI12_prval_ratio(disease2emr_count,
                                                    norm_prval_method,
                                                    prval_ratio_PHI1,
                                                    prval_ratio_PHI2)

        norm_factors["PHI1"] = 1.0 / prval_ratio_PHI1
        norm_factors["PHI2"] = 1.0 / prval_ratio_PHI2
        norm_factors["PHI12"] = 1.0 / prval_ratio_PHI12

    return norm_factors

def __get_PHI12_prval_ratio(disease2emr_count,
                            norm_prval_method,
                            prval_rat_PHI1,
                            prval_rat_PHI2):

    """Derive and return ratio of EMR prevalence of D1andD2 to the population
    prevalence of D1andD2 using specified norm_prval_method."""

    if norm_prval_method == "max":
        prval_rat_PHI12 = max(prval_rat_PHI1, prval_rat_PHI2)

    elif norm_prval_method == "min":
        prval_rat_PHI12 = min(prval_rat_PHI1, prval_rat_PHI2)

    elif norm_prval_method == "avg":
        prval_rat_PHI12 = (prval_rat_PHI1 + prval_rat_PHI2) / 2.0

    elif norm_prval_method == "wts_avg":
        D1_count = disease2emr_count[D1]
        D2_count = disease2emr_count[D2]
        D1_wts = float(D1_count) / (D1_count + D2_count)
        D2_wts = float(D2_count) / (D1_count + D2_count)
        prval_rat_PHI12 = D1_wts * prval_rat_PHI1 + D2_wts * prval_rat_PHI2

    elif norm_prval_method == "sum":
        # This serves as an upper bound to independent assumption.
        prval_rat_PHI12 = prval_rat_PHI1 + prval_rat_PHI2

    elif norm_prval_method == "independent":
        # Pr(V | D1andD2) = 1.0 - ([1 - Pr(V | D2notD1)] *
        #                          [1 - Pr(V | D1notD2)])
        # visit_rate (rate that patients visit Stanford Hospital is not known!)
        #
        # prval_rat_PHI12 = (prval_rat_PHI1 + prval_rat_PHI2 -
        #                    prval_rat_PHI1 * prval_rat_PHI2 * visit_rate)
        raise NotImplementedError("""Unable to compute Pr(V | D1andD2) 
                                  under independent assumption since 
                                  Stanford hospital visit rate not known!""")
    else:
        raise ValueError("Invalid norm_prval_method %s" % norm_prval_method)

    return prval_rat_PHI12

def __compute_rzhetsky_prval_ratio(D1, D2,
                                   disease2emr_count,
                                   total_patient_count,
                                   population_prval,
                                   norm_prval_method):
    """Compute and the return ratio of EMR prevalence of disease to population 
    prevalence of disease using the method in Rzhetsky et. al. 2007.

    In Rzhetsky et al. 2007, the raw EMR prevalence is adjusted by identical
    factor for every disease. This factor was determine by the ratio EMR
    prevalence to general population of Schizophrenia (Bipolar was also
    considered and shown to give very similar ratio in their paper.

    From Rzhetsky et al:
                                                                Estimated
                                Raw EMR         Adjusted        Population
                                % Prevalence    % Prevalence    % Prevalence
        Bipolar                 0.822           1.230           ~1.20
        Schizophrenia           0.747           1.119           ~1.10

    Note that bipolar population prevalence used in Rzhetsky (1.2%) is 
    does not match with the value reported at the CDC website (http://www.cdc.
    gov/mentalhealth/basics/burden.htm):

        "The National Comorbidity Study reported a lifetime prevalence of
         nearly 4% for bipolar disease."
    """

    D = "Schizophrenia"

    rzhetsky_emr_prval = float(disease2emr_count[D]) / total_patient_count
    rzhetsky_pop_prval = population_prval[D]

    rzhetsky_prval_ratio = rzhetsky_emr_prval / rzhetsky_pop_prval

    return rzhetsky_prval_ratio

def __import_prevalence_file(infile):
    """Import data on prevalence of disease in general population.

    Prevalence value in the infile should be in percentage. Will divide
    by 100.0 to convert to decimal format.

    Parameters
    ----------
    infile: string
        Location of the prevalence data file.

    Returns
    -------
    population_prval: dictionary
        A dictionary mapping disease name to the prevalence of disease in
        general population (in decimal representation).

    """

    population_prval = dict()

    for line in open(infile):

        line = line.replace("\n", "")
        line = line.replace("\r", "")

        if line == "": continue  # empty line
        if line[0] == "#": continue  # comments line

        cols = line.split(",")

        if len(cols) != 2:
            raise IOError("line \"%s\" should have 2 CSV fields " % line)

        disease = cols[0]

        prevalence = float(cols[1])
        prevalence = prevalence / 100.00  # convert from percent to decimal

        if prevalence <= 0.0 or  prevalence > 1.00:
            raise ValueError("prevalence %s is out of range" % prevalence)

        if disease in population_prval:
            raise ValueError("duplicated disease %s." % disease)

        population_prval[disease] = prevalence

    return population_prval
