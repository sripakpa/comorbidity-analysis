INTRODUCTION:

    The python package contains scripts for data mining of the Electronic
    Medical Records (EMR) to predict predict diseases genetic overlaps and 
    comorbidities.
    
    The models used here is an extension of the probabilistic model proposed
    in Rzhetsky et. al. 2007 [1].

    This package is currently being used to mine EMR data obtained from
    Stanford Hospital through the Stanford Translational Research Integrated
    Database Environment (STRIDE) [2]. The EMR dataset consist of anoynmized
    statistics of over 1 million patient records for 161 diseases.

    This package is a work in-progress. Questions that are actively being
    investigated include:

    (1) How accurately can the EMR dataset predict diseases comorbidities
        when assessed on a benchmark of 300 known comorbidities curated from
        from the literature?

    (2) Differing rates of hospitalization for each diseases can lead to
        bias in the EMR counts. This is a well known problem in
        epidemiological study called the Berkson's bias [3, 4]. How strongly
        does Berkson's bias effect the accuracy of the diseases comorbidities
        predictions?

    (3) Can the effect of Berkson's bias be reduced by normalizing the EMR
        dataset using population prevalence data? Can machine learning
        approaches be used to "learn" the rates of hospitalization and improve
        prediction accuracy?

DATA:

    (1) data/test_EMR_dataset.csv

        The actual STRIDE EMR dataset is not included in this repository due
        to the sensitive nature of the data. Instead we include a test EMR
        dataset consisting of 48,000 "mock" patient medical records for 6
        diseases.

    (2) data/disease-list.txt

        A list of the 161 diseases investigated in this study.

    (3) data/code-to-disease.csv

        A CSV file mapping ICD-9 codes to disease names.

    (4) data/disease-prevalence.csv

        A list of disease prevalence estimates in the general population.

DESCRIPTIONS:

    Computing P-value
    -----------------

    (1) p_value.py

        Computes the statistical significance (p-value) of observing the EMR
        dataset for a pair of diseases according to the genetic penetrance
        models proposed in Rzhetsky et al. 2007 [1].

        Usage: 
            python p_value.py -i <emr_data_file> \
                              --d1 <1st_disease_name> \
                              --d2 <2nd_disease_name>

    Plotting/Data visualization
    ---------------------------

    (1) plot_age_of_onset.py


        Plots the age-of-onset distribution of diseases using the data
        derived from the EMR dataset.

        Usage: 
              python plot_age_of_onset.py -i <emr_data_file> \
                                          --disease <disease_name>

    (2) plot_joint_age_of_onset.py

        Plots the joint age-of-onset distribution for a pair of diseases using
        the data derived from the EMR dataset.

        Usage: 
            python plot_joint_age_of_onsets.py -i <emr_data_file> \
                                               --d1 <1st_disease_name> \
                                               --d2 <2nd_disease_name>

    (3) plot_joint_final_age.py

        Plots the count of patients as a distribution of their final age value. 
        The patients are seperated by their 2 disease phenotype status and the
        count are computed seperately for each phenotype status.

        Usage:
            python plot_joint_final_age.py -i <emr_data_file> \
                                           --d1 <1st_disease_name> \
                                           --d2 <2nd_disease_name>

    (4) plot_genetic_penetrance.py

        Plots the age-integrated phenotype probability for a pair of
        diseases D1 and D2 using the genetic penetrance models proposed by
        Rzhetsky et al. 2007.

        Usage:
            python plot_genetic_penetrance.py

    Tests
    -----

    (1) test_emr_database.py

        Tests the functionality of the emr_database_class.

        Usage:
            python emr_database_test.py -i <emr_data_file>

    (2) test_optimize_log_likelihood.py

        Test the optimization of the log_likelihood function for explaining the
        EMR database using the genetic penetrance models proposed by Rzhetsky
        et al. 2007.

        Usage:
            python test_optimize_log_likelihood.py -i <emr_data_file> \
                                                   --d1 <1st_disease> \
                                                   --d2 <2nd_disease>

AUTHOR:

    Parin Sripakdeevong <sripakpa@stanford.edu>

COLLABORATORS:

    Dr. Steven Bagley (Stanford University)

    Prof. Russ Altman (Stanford University)

REFERENCES:

    [1] Rzhetsky, A., Wajngurt, D., Park, N. & Zheng, T. Probing genetic 
        overlap among complex human phenotypes. Proc Natl Acad Sci U S A 
        104, 11694-9 (2007).

    [2] https://clinicalinformatics.stanford.edu/research/stride.html

    [3] Berkson, J. Limitations of the application of fourfold table analysis
        to hospital data. Biometrics 2, 47-53 (1946).

    [4] http://en.wikipedia.org/wiki/Berkson's_paradox
