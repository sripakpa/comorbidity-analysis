#!/usr/bin/env python
"""
SYNOPSIS
    This script tests the functionality of the emr_database_class.

DESCRIPTION
    This script tests the import and query functionality of the
    emr_database_class. 

    Perform test query on specific disease and disease pairs, including
    the gender and ethnicity filters.

EXAMPLES

    # Standard
    python test_emr_database.py -i data/test_EMR_dataset.csv

    # Specify diseases_file and code2disease_file -v -c 
    python test_emr_database.py -i data/test_EMR_dataset.csv \
           --diseases_file data/disease-list.txt \
           --code2disease_file data/code-to-disease.csv 

AUTHOR
    Parin Sripakdeevong <sripakpa@stanford.edu>"""

import sys
import time
import optparse

from emr_database_class import EMRDatabase

def main():

    global options

    database = EMRDatabase(options.verbose)

    database.import_data(options.emr_data_file,
                         options.diseases_file,
                         options.code2disease_file)

    diseases = ["Alzheimer's disease", "Attention deficit", "Autism", 
                 "Breast cancer (female)", "Epilepsy", "Schizophrenia"]

    D1 = "Autism"
    D2 = "Epilepsy"
    D3 = "Breast cancer (female)"

    # Query for patients that have D1
    D1_patients = database.query_emr_data(D1)

    # Query for patients that have D2
    D2_patients = database.query_emr_data(D2)

    # Query for patients that have both D1 and D2
    D1andD2_patients = database.query_emr_data([D1, D2])

    # Query for patients that have either D1 or D2
    D1orD2_patients = database.query_emr_data([D1, D2], OR_match = True) 

    # Query for patients that have D1 and not D2
    D1notD2_patients = database.query_emr_data([D1, "not " + D2])

    # Query for patients that have D2 and not D1
    D2notD1_patients = database.query_emr_data(["not " + D1, D2])

    # Query for patients that have neither D2 nor D1
    noD1D2_patients = database.query_emr_data(["not " + D1, "not " + D2])

    # Query for male patients with female Breast cancer
    D3_male_patients = database.query_emr_data(D3 , gender_filters = "M")

    # Query for female patients with female Breast cancer
    D3_female_patients = database.query_emr_data(D3, gender_filters = "F")

    print "Test EMR_database queries:"
    print "%-6d patients with %s" %(len(D1_patients), D1)
    print "%-6d patients with %s" %(len(D2_patients), D2)
    print "%-6d patients with %s and %s" %(len(D1andD2_patients), D1, D2)
    print "%-6d patients with %s or %s" %(len(D1orD2_patients), D1, D2)
    print "%-6d patients with %s and no %s" %(len(D1notD2_patients), D1, D2)
    print "%-6d patients with %s and no %s" %(len(D2notD1_patients), D2, D1)
    print "%-6d patients with no %s and no %s" %(len(noD1D2_patients), D1, D2)
    print "%-6d male patients with %s" %(len(D3_male_patients), D3)
    print "%-6d female patients with %s" %(len(D3_female_patients), D3)

    if options.verbose:
        print "Patients with %s:" % D1
        for patient in D1_patients:
            print "patient %s --> %s" % (patient, D1_patients[patient])
        print

        print "Patients with %s:" % D2
        for patient in D2_patients:
            print "patient %s --> %s" % (patient, D2_patients[patient])
        print

        print "Patients with %s and %s:" % (D1, D2)
        for patient in D1andD2_patients:
            print "patient %s --> %s" % (patient, D1andD2_patients[patient])
        print

        print "Patients with %s or %s:" % (D1, D2)
        for patient in D1orD2_patients:
            print "patient %s --> %s" % (patient, D1orD2_patients[patient])
        print

        print "Patients with %s and no %s:" % (D2, D1)
        for patient in D2notD1_patients:
            print "patient %s --> %s" % (patient, D2notD1_patients[patient])
        print

        print "Patients with %s and no %s:" % (D1, D2)
        for patient in D1notD2_patients:
            print "patient %s --> %s" % (patient, D1notD2_patients[patient])
        print

        print "Patients with no %s and no %s:" % (D1, D2)
        for patient in noD1D2_patients:
            print "patient %s --> %s" % (patient, noD1D2_patients[patient])
        print

        print "Male patients with %s:" % (D3)
        for patient in D3_male_patients:
            print "patient %s --> %s" % (patient, D3_male_patients[patient])
        print

        print "Female patients with %s:" % (D3)
        for patient in D3_female_patients:
            print "patient %s --> %s" % (patient, D3_female_patients[patient])
        print


    # Find occurrences of 6 diseases.
    print
    print "Occurrences of the 6 diseases:"
    for n1, D1 in enumerate(diseases):
        D1_patients = database.query_emr_data(D1)
        print "%-6d patients with %s" %(len(D1_patients), D1)
        if options.verbose:
            print "Patients with %s:" % D1
            for patient in D1_patients:
                print "patient %s --> %s" % (patient, D1_patients[patient])
            print

    # Find comorbidities between the 6 diseases.
    print
    print "Comorbidities between the 6 diseases:"


    for n1, D1 in enumerate(diseases):
        for n2, D2 in enumerate(diseases):
            if n1 >= n2: continue

            D1andD2_patients = database.query_emr_data([D1, D2])

            print ("%-6d patients with %s and %s" 
                   %(len(D1andD2_patients), D1, D2))

            if options.verbose:
                print "Patients with both %s and %s:" % (D1, D2)
                for patient in D1andD2_patients:
                    print ("patient %s --> %s" % (patient, 
                           D1andD2_patients[patient]))
                print

if __name__ == '__main__':

    start_time = time.time()

    usage = 'python emr_database_test.py -i <emr_data_file>'
    parser = optparse.OptionParser(usage=usage + globals()['__doc__'])

    parser.add_option("-i", "--emr_data",
                      action="store", type="string", dest="emr_data_file",
                      help="RAW EMR data file")

    parser.add_option("--diseases_file", action="store",
                      default="data/disease-list.txt", type="string",
                      dest="diseases_file",
                      help="file contain list of diseases.")

    parser.add_option("--code2disease_file", action="store",
                      default="data/code-to-disease.csv", type="string",
                      dest="code2disease_file",
                      help="CSV file mapping ICD9 code to disease.")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      dest="verbose", help="verbose output.")

    parser.add_option("--no_verbose", action="store_false", default=False,
                      dest="verbose", help="no verbose output.")


    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error("leftover arguments=%s" % args)

    if not options.emr_data_file:
        parser.error("option -i required")

    if options.verbose:
        print "-" * 50
        print time.asctime()
        print "emr_data_file: %s" % options.emr_data_file
        print "diseases_file: %s" % options.diseases_file
        print "code2disease_file: %s" % options.code2disease_file
        print "-" * 50

    main()
    #if options.verbose:
    print time.asctime(),
    print ' | total_time = %.3f secs' % (time.time() - start_time)

    sys.exit(0)
