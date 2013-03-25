#!/usr/bin/env python

import sys
import copy

class EMRDatabase(object):
    """EMRDatabase class

    This class handles importing and querying Electronic Medical Records
    (EMRs) data.

    Import and store (1) EMR_data, (2) disease list, and (3) ICD-9 to disease
    mappings.

    Support fast query of EMR data by diseases, gender,and ethnicity
    citeria.

    Parameters
    ----------
    verbose: bool, optional
        Verbose output.

    Attributes
    ----------
    diseases : list of string
        A list of disease names (string type).

    code2disease : dictionary
        A dictionary mapping ICD-9 code (string) to disease name (string).

    disease2codes : dictionary
        A dictionary mapping disease name (string) to a list of ICD-9 codes
        (string)

    non_disease_data : two layer dictionary
        A nested dict() containing all non disease-related data about the
        patients. The dictionary key is the patient_ID unique to each
        individual patient.

        Each non_disease_data[patient_ID] is itself a dict() with the
        following keys:
            "age": Age of patient at most recent visit.
            "gender": Gender of the patient
            "ethnicity": Ethnicity of the patient.

    disease2patients : three layer dictionary
        1st layer: disease2patients is a dict() with disease (D) as its
            keys.
        2nd layer: disease2patients[D] is dict() with the patient's ID
            as its key. Only patient's ID of patients that have disease D
            are stored in the dictionary.
        3nd layer: disease2patients[D][ID] is a dict() with the following
            keys:
                'age_of_onset': patient's age at the time he or she was
                    first diagnosed with the disease. This is approximated
                    using the patient's age during the hospital visit that
                    the associated ICD-code first appear.

    gender_str2code: dictionary
        A dictionary mapping gender string (e.g. "MALE", "FEMALE") to 
        one-letter gender code ("M" or "F").

    ethnicity_str2code: dictionary
        A dictionary mapping ethnicity string (e.g. "WHITE", "ASIAN" to 
        one-letter ethnicity code (e.g. "W", "A").

    emr_stat : dictionary
        Basic statistic about the imported EMR data including
        total entries, ignored entires, unique patient count, and unique
        disease instance count.
 """

    def __init__(self, verbose=False):
        """Instantiate variables in EMRDataBase class."""
        self.verbose = verbose

        self.diseases = None
        self.code2disease = None
        self.disease2codes = None
        self.non_disease_data = None
        self.disease2patients = None
        self.gender_str2code = None
        self.ethnicity_str2code = None
        self.emr_stat = None

        self.__init_gender_str2code()
        self.__init_ethnicity_str2code()

    # Public methods
    def import_data(self, emr_data_file, diseases_file, code2disease_file):
        """Import diseases, code2disease mapping, and EMR data from file."""

        self.__import_diseases(diseases_file)

        if self.verbose: self.__print_diseases()

        self.__import_code2disease(code2disease_file)

        self.__map_disease2codes()

        if self.verbose: self.__print_disease2codes()

        # Responsible for initializing and filling up: 
        #  (1) self.non_disease_data
        #  (2) self.disease2patients
        #  (3) self.emr_stat
        self.__import_EMR_data(emr_data_file)

        if self.verbose: self.__print_non_disease_data()

        self.__print_data_summary(emr_data_file,
                                  diseases_file,
                                  code2disease_file)


    def get_diseases(self):
        """Return a deep copy of the diseases object."""

        return copy.deepcopy(self.diseases)

    def get_disease2count(self):
        """Return a dictionary mapping disease name to its count."""

        disease2count = dict()

        for D in self.diseases:

            patient_counts = len(self.disease2patients[D])

            if D in disease2count:
                raise ValueError("Duplicated disease %s." % D)

            disease2count[D] = patient_counts

        return disease2count

    def get_tot_patient_count(self):
        """Return the total number of unique patients in the EMR dataset."""

        return len(self.non_disease_data)

    def query_emr_data(self,
                       diseases,
                       OR_match = False,
                       gender_filters = None,
                       ethnicity_filters = None):

        """Query and return matching EMR data of patients base on diseases,
        gender and and ethnicity citeria.

        (1) By default (OR_match is False) and query will select patients that
        have all the specified diseases.

        (2) If OR_match is True, then the query is relaxed and instead select
        for patients that have any of the specified diseases.

        (3) If the substring "not " appears at the beginning of a
        disease (e.g. "not Alzheimer's disease"), then the patients WITHOUT
        the disease is selected instead. 
        For example: ["Autism", "not Alzheimer's disease"] would select 
        for patients that have Autism but not Alzheimer's disease.

        Parameters
        ----------
        disease_list : string or list of string
            Return patients based on the specified diseases.

        OR_match: bool, optional
            False: select patients that have all of the specified diseases.
            True: select patients that have any of the specified diseases.

        gender_filters : {"M", ["M"], "F", ["F"], ["M", "F"], None}, optional
            Gender types to be selected:
            (1) "M" or ["M"] will select for only male patients.
            (2) "F" or ["F"] will  select for only female patients
            (3) ["M", "F"] will select both male and female
                patients.
            (4) None (default): no filter is applied and patients
                of all genders will be selected.
            Note that ["M", "F"] and None might give slightly different results
            if undefined or ambigious gender are allowed. 

        ethnicity_filters : string or set of string, optional
            Ethnicity types to be selected. Patient that match one of the 
            specified ethnicity type are selected. In the default case (None),
            no filter will be applied and patients of all ethnicity are
            selected.

        Returns
        -------
        EMR_data : two layer dictionary
            EMR data of all patients that satisfy the query. The patient's
            ID serves as the dictionary key and each EMR_data[ID] contains
            the following keys-value pair.
                'age': patient's age in years
                'ethnicity': patient's ethnicity.
                'gender': patient's gender
                'age_of_onsets': a list of age-of-onsets, one value for each of
                    the specified diseases.
                    For example, EMR_data[ID]['age_of_onsets'][0] is the
                    age-of-onset of diseases[0] for patient with ID.
                    If the patient never have the disease, that the
                    corresponding age-of-onset is set to None

        Notes
        -----
        Create deep copy, so that subsequent modifications of the EMR_data
        will not effect the private data stored in this class.
        """

        if OR_match: # match diseases by OR instead of AND condition.

            EMR_data_list = []
            OR_match = False

            for disease in diseases:

                # Call itself but with single disease and OR_match = False
                emr_data = self.query_emr_data(disease, OR_match,
                                               gender_filters, 
                                               ethnicity_filters)

                EMR_data_list.append(emr_data)

            return self.__combine_EMR_data(EMR_data_list)

        # Careful not modify the variables passed in the function.
        diseases = copy.deepcopy(diseases)
        gender_filters = copy.deepcopy(gender_filters)
        ethnicity_filters = copy.deepcopy(ethnicity_filters)

        # Check the input variables are valid.
        if gender_filters is not None:
            self.__validate_gender(gender_filters)

        if ethnicity_filters is not None: 
            self.__validate_ethnicity(ethnicity_filters)

        # Convert string to list of string.
        if isinstance(diseases, str): 
            diseases = [diseases]
        if isinstance(gender_filters, str): 
            gender_filters = [gender_filters]
        if isinstance(ethnicity_filters, str): 
            ethnicity_filters = [ethnicity_filters]

        # Keep track of whether to select for patient WITH or WITHOUT the
        # disease.
        negate_D = [] 

        for n in xrange(len(diseases)):
            if diseases[n].startswith("not "):
                negate_D.append(True)
                diseases[n] = diseases[n][4:]  # remove "not ".
            else:
                negate_D.append(False)

        self.__validate_disease(diseases)

        # Select the smallest patient IDs dictionary that still contain all
        # patients with the specified diseases.
        IDs_dict = self.__find_smallest_IDs_dict(diseases, negate_D)

        return  self.__fill_up_EMR_data(IDs_dict, diseases, negate_D,
                                        gender_filters, ethnicity_filters)


    # Private methods
    def __import_diseases(self, infile):
        """Import list of diseases from text file.

        Example lines:
            Actinomycosis
            Acute leukemia
            Acute promyelocytic leukemia
        """

        if infile == None:
            raise RuntimeError("No diseases_file specified!")

        self.diseases = list()

        for line in open(infile):

            line = line.replace("\n", "")
            line = line.replace("\r", "")

            if line == "": continue  # empty line
            if line[0] == "#": continue  # comments line

            self.diseases.append(line)

    def __import_code2disease(self, infile):
        """Import ICD-9 code to disease mapping from CSV text file and store
        data in a python dictionary.

        Examples lines:
            "039.8","Actinomycosis"
            "039.9","Actinomycosis"
            "208.0","Acute leukemia"
            "205.0","Acute promyelocytic leukemia"

        Notes
        -----
            One disease can map to multiple ICD-9 code, but each ICD-9 code
            can only map to one disease.
        """

        if infile == None:
            raise RuntimeError("No code2disease_file specified!")

        self.code2disease = dict()

        for line in open(infile):

            line = line.replace("\n", "")
            line = line.replace("\r", "")

            if line == "": continue  # empty line
            if line[0] == "#": continue  # comments line

            maxsplit = 1
            cols = line.split(",", maxsplit)

            if len(cols) != 2:
                raise IOError("line \"%s\" should have 2 CSV fields " % line)

            code = cols[0][1:-1]  # remove the "" wrapping.

            disease = cols[1][1:-1]  # remove the "" wrapping.

            if code in self.code2disease:
                raise ValueError("duplicated ICD-9 code %s." % code)

            self.code2disease[code] = disease

    def __map_disease2codes(self):
        """Create mapping disease to list of ICD-9 codes."""

        self.disease2codes = dict()

        for code in self.code2disease:

            disease = self.code2disease[code]

            if not self.disease2codes.has_key(disease):
                self.disease2codes[disease] = list()

            self.disease2codes[disease].append(code)

        for D in self.diseases:
            if D not in self.disease2codes:
                raise KeyError("disease \"%s\" missing in disease2codes" % D)


    def __import_EMR_data(self, data_file):
        """import EMR database from file

        Responsible for intializing and filling up:
          (1) self.non_disease_data
          (2) self.disease2patients
          (3) self.emr_stat

        Here is the (current) format of the import data (CSV format):
          (1) Deidentified patient ID.
          (2) Patient's current age
          (2) Patient's gender.
          (3) Patient's race/ethnicity.
          (4) ICD-9 code of disease.
          (5) Visit-Age (more strictly, the age of first appearance)
          (6) Comments  (this line will be ignored)

        Will assume that the Patient's current age and visit age is
        an integer (in years).

        Parameters
        ----------
         data_file : location of the file containing the input data.
            The file should be a CSV text file where each row correspond
            to a EMR instance (one patient during one visit with one 
            ICD code list).

        Notes
        -----
        Will ignore entry with Patient's gender is "unknown".

        Will assumes that the gender and ethnicity of a patient
        cannot change between visits.
        """

        if data_file == None:
            raise RuntimeError("No EMR_data_file specified!")

        self.non_disease_data = dict()

        # Initialize disease2patients.
        self.disease2patients = dict()
        for D in self.diseases: self.disease2patients[D] = dict()

        # Initialize emr_stat.
        self.emr_stat = {'total_entries': 0, 'ignored_entries': 0,
                         'patient_count': 0, 'disease_count': 0}

        for line in open(data_file):

            line = line.replace("\n", "")
            line = line.replace("\r", "")

            if line == "": continue  # empty line
            if line[0] == "#": continue  # comments line

            cols = line.split(",")

            if len(cols) < 6:
                raise IOError("line %s should at least 6 CSV fields." % line)

            ID = cols[0]
            current_age = int(cols[1])
            gender = self.__parse_gender(cols[2])
            ethnicity = self.__parse_ethnicity(cols[3])
            ICD_9 = cols[4]
            visit_age = int(cols[5])

            # In rare cases the recorded visit age is greater than current_age. 
            # For example, patient #16161723 has an recorded visit_age of 90
            # but a current_age of 70.
            current_age = max(current_age, visit_age)

            self.emr_stat['total_entries'] += 1

            # Ignore lines with missing and unrecognized fields.
            if gender == "unknown":
                self.emr_stat['ignored_entries'] += 1
                if self.verbose: 
                    print "WARNING: Ignoring EMR data line: \"%s\" " % line,
                    print "REASON: Missing gender field.\n"
                    print
                continue

            if ICD_9 not in self.code2disease:
                self.emr_stat['ignored_entries'] += 1
                if self.verbose: 
                    print "WARNING: Ignoring EMR data line: \"%s\" " % line,
                    print "REASON: ICD_9 code %s not recognized." % ICD_9
                    print
                continue

            D = self.code2disease[ICD_9]

            if D not in self.diseases:
                self.emr_stat['ignored_entries'] += 1
                if self.verbose: 
                    print "WARNING: Ignoring EMR data line: \"%s\" " % line,
                    print "REASON: Disorder %s not reconginized." % D
                    print
                continue

            self.__validate_gender(gender)

            self.__validate_ethnicity(ethnicity)

            self.__validate_ICD_9(ICD_9)

            self.__validate_disease(D)

            self.__update_non_disease_data(ID, gender, ethnicity, current_age)

            self.__update_disease2patients(ID, D, visit_age)

    def __update_disease2patients(self, ID, D, visit_age):
        """Update or add new data to the disease2patients dictionary."""

        self.__validate_disease(D)

        # Update an existing record.
        if ID in self.disease2patients[D]:

            # Age of onset of the disease is approximated using the patient's
            # age during the hospital visit that the associated ICD-code first
            # appear.
            if self.disease2patients[D][ID]['age_of_onset'] > visit_age:
                self.disease2patients[D][ID]['age_of_onset'] = visit_age

        else: # Create a new record.
            self.emr_stat['disease_count'] += 1
            self.disease2patients[D][ID] = {'age_of_onset': visit_age}

    def __update_non_disease_data(self, ID, gender, ethnicity, current_age):
        """Update ord add new data to the non_disease_data dictionary."""

        # Update an existing record.
        if ID in self.non_disease_data:

            # In rare cases, the current_age of same patient could
            # could vary from row to row. Use the highest age.
            if current_age > self.non_disease_data[ID]['age']:
                self.non_disease_data[ID]['age'] = current_age

            # ensure that patient's gender did not change!
            if gender != self.non_disease_data[ID]['gender']:
                raise AssertionError("gender of patient %s changed." % ID) 

            # ensure that patient's ethnicity did not change!
            if ethnicity != self.non_disease_data[ID]['ethnicity']:
                raise AssertionError("ethnicity of patient %s changed." % ID) 

        else: # A new record
            self.emr_stat['patient_count'] += 1

            self.non_disease_data[ID] = {'age': current_age,
                                         'gender': gender, 
                                         'ethnicity': ethnicity}

    def __init_gender_str2code(self):
        """Initialize an dict mapping gender string to one-letter code."""

        self.gender_str2code = {}

        self.gender_str2code["M"] = "M"
        self.gender_str2code["male"] = "M"
        self.gender_str2code["Male"] = "M"
        self.gender_str2code["MALE"] = "M"

        self.gender_str2code["F"] = "F"
        self.gender_str2code["female"] = "F"
        self.gender_str2code["Female"] = "F"
        self.gender_str2code["FEMALE"] = "F"

        self.gender_str2code["UNKNOWN"] = "unknown"

    def __parse_gender(self, G_str):
        """Convert the input gender string into a gender code."""

        if G_str not in self.gender_str2code:
            raise KeyError("Invalid gender_string %s" % G_str)

        return self.gender_str2code[G_str]

    def __validate_gender(self, G_codes):
        """Validate the gender code(s).

        Notes
        -----
        Valid gender codes are:
            Code        Explanation
            "M"           Male   
            "F"           Female
        """

        valid_codes = set(["M", "F"])

        if isinstance(G_codes, str): G_codes = [G_codes]

        for G in G_codes:
            if G not in valid_codes:
                raise KeyError("invalid gender_code \"%s\"." % G)

    def __init_ethnicity_str2code(self):
        """Initialize an dict mapping ethnicity string to one-letter code.

        Notes
        -----
        Right now deal with the input ethnicity string from the STRIDE
        dataset. The possibilities encountered are:
            ethnicity string            Code
                ASIAN                   "A"
                WHITE                   "W"
                BLACK                   "B"
                OTHER                   "O"
                PACIFIC ISLANDER        "P"
                NATIVE AMERICAN         "I"
                UNKNOWN                 "U"
        """

        self.ethnicity_str2code = {}

        self.ethnicity_str2code["ASIAN"] = "A"
        self.ethnicity_str2code["WHITE"] = "W"
        self.ethnicity_str2code["BLACK"] = "B"
        self.ethnicity_str2code["OTHER"] = "O"
        self.ethnicity_str2code["PACIFIC ISLANDER"] = "P"
        self.ethnicity_str2code["NATIVE AMERICAN"] = "I"
        self.ethnicity_str2code["UNKNOWN"] = "U"

    def __parse_ethnicity(self, E_str):
        """Convert the input ethnicity string into a gender code."""

        if E_str not in self.ethnicity_str2code:
            raise KeyError("Invalid ethnicity_string %s" % E_str)

        return self.ethnicity_str2code[E_str]

    def __validate_ethnicity(self, E_codes):
        """Validate the ethnicity code(s).

        Notes
        -----
        Valid ethnicity codes are:
            Code    Explanation
            "A"       Asian (including Pacific Islanders)        
            "B"       Black/non-Hispanic
            "D"       Response regarding ethnicity declinced
            "E"       Oriental
            "H"       White-Hispanic
            "I"       American Indian (Native American) or Alaskan
            "M"       Middle eastern
            "N"       Indian from India
            "L"       Latin American
            "O"       Other
            "P"       Pacific Islanders
            "U"       Unknown
            "W"       White/non-Hispanic
            "X"       Black

        These code based on the Columbia EMR database (Rzhetsky et. al. 2007).
        """

        valid_codes = set(["A", "B", "D", "E", "H", "I", "M", 
                           "N", "L", "O", "P", "U", "W", "X"])

        if isinstance(E_codes, str): E_codes = [E_codes]

        for E in E_codes:
            if E not in valid_codes: 
                raise ValueError("invalid ethnicity code %s." % E)

    def __validate_ICD_9(self, ICD_9_codes):
        """Ensure that ICD_9 code(s) are in code2disease."""

        if isinstance(ICD_9_codes, str): ICD_9_codes = [ICD_9_codes]

        for I in ICD_9_codes:
            if I not in self.code2disease:
                raise KeyError("ICD-9 code \"%s\" missing in" +
                               "code2disease" % I)

    def __validate_disease(self, disease_names):
        """Ensure that disease name(s) are in diseases and 
           disease2patients."""

        if isinstance(disease_names, str): disease_names = [disease_names]

        for D in disease_names:

            if D not in self.diseases:
                raise ValueError("Disorder \"%s\" missing in diseases" % D)

            if D not in self.disease2patients:
                raise KeyError("Disorder \"%s\" missing in" +
                                + "disease2patients" % D)

    def __print_diseases(self):
        """Print diseases #5, #10, #15, #20 and #25 to STDOUT."""

        print "diseases (#5, #10, #15, #20, #25):"
        for n in range(5,30,5):
            if n < len(self.diseases): 
                print "%2d %s" % (n, self.diseases[n])
        print

    def __print_disease2codes(self):
        """Print disease2codes for diseases #5, #10, #15, #20, #25 to STDOUT.
        """

        print "disease to ICD-9 codes (#5, #10, #15, #20, #25):"
        for n in range(5,30,5):
            if n < len(self.diseases):
                D = self.diseases[n]
                print "%s --> %s" % (D, self.disease2codes[D])
        print

    def __print_non_disease_data(self):
        """Print non disease-related data for five psuedo-random patients
           from the EMR database."""

        count = 0

        print "5 random patients (of %d) " % len(self.non_disease_data),
        print "from EMR database (non disease-related data):" 
        for ID in self.non_disease_data:
            count += 1
            print "ID %s --> %s" % (ID, self.non_disease_data[ID])
            if count > 5: break
        print

    def __print_data_summary(self, 
                             emr_data_file,
                             diseases_file,
                             code2disease_file):
        """Print summary of the imported data."""

        print "From %s:" % diseases_file
        print "(1) Imported %d diseases" % len(self.diseases)
        print

        print "From %s:" % code2disease_file
        print "(1) Imported %d unique ICD-9 codes " % len(self.code2disease)
        print

        # short hand.
        stat = self.emr_stat

        print "From %s:" % emr_data_file
        print "(1) Imported %d total entries" % stat['total_entries'],
        print "of which %d entries were ignored." % stat['ignored_entries']
        print "(2) Imported %d unique patients" % stat['patient_count']
        print "(3) Imported %d unique diseases" % stat['disease_count']
        print

    def __combine_EMR_data(self, in_EMR_data_list):
        """Combine multiple EMR_data into a single EMR_data.

        Each of the input EMR_data in in_EMR_data_list is assume to contain
        age-of-onset information for a single disease.

        The output EMR_data will contain age-of-onset of all diseases found in
        in_EMR_data_list. Specifically age-of-onsets[i] will contain the
        age-of-onset information from in_EMR_data_list[i].
        """

        out_EMR_data = dict()

        num_diseases = len(in_EMR_data_list)

        for i, in_EMR_data in enumerate(in_EMR_data_list):

            for ID in in_EMR_data:

                if len(in_EMR_data[ID]['age_of_onsets']) != 1:
                    raise AssertionError("age_of_onsets' length should be 1.")

                if ID not in out_EMR_data: 
                    # New patient. Add patient's EMR data to out_EMR_data.
                    out_EMR_data[ID] = copy.deepcopy(in_EMR_data[ID])
                    out_EMR_data[ID]['age_of_onsets'] = [None] * num_diseases

                age_of_onset_i = in_EMR_data[ID]['age_of_onsets'][0]
                out_EMR_data[ID]['age_of_onsets'][i] = age_of_onset_i

        return out_EMR_data

    def __find_smallest_IDs_dict(self, diseases, negate_D):
        """Select the smallest patient IDs dictionary to iterate over.

        Goal is find smallest patient IDs dictionary that still all the
        patients with the specified diseases.

        The most general case, is to select the non_disease_data dict, which
        contain all patients ID, but this dictionary can potentially contain a
        huge number (millions) of patient IDs.

        A smaller dictionary would be disease2patients[D], which only
        contain patients with the disease D.

        NOTES
        -----
        negate_D is a list of boolean. If negate_ID[i] is false, 
        then will select for patients with diseases[i]. If negate_ID[i] is
        true, then will instead select patients without diseases[i]

        """

        IDs_dict = self.non_disease_data 

        # If possible, find the smallest dictionary to iterate over.
        for n, D in enumerate(diseases):

            if negate_D[n]: continue

            if D not in self.disease2patients:
                raise KeyError("invalid disease %s queried." % D)

            if len(IDs_dict) > len(self.disease2patients[D]):
                IDs_dict = self.disease2patients[D]

        return IDs_dict

    def __fill_up_EMR_data(self, IDs_dict, diseases, negate_D, gender_filters,
                           ethnicity_filters):
        """Initiate and fill up the EMR data.

        NOTES
        -----
        negate_D is a list of boolean. If negate_ID[i] is false, 
        then will select for patients with diseases[i]. If negate_ID[i] is
        true, then will instead select patients without diseases[i].
        """

       # Initiate and fill up the EMR data
        EMR_data = dict()

        for ID in IDs_dict:

            # gender filters
            if gender_filters is not None:
                gender = self.non_disease_data[ID]['gender'] 
                if gender not in gender_filters: continue

            # ethnicity filters
            if ethnicity_filters is not None:
                ethnicity = self.non_disease_data[ID]['ethnicity']
                if ethnicity not in ethicity_filters: continue

            # select for desired diseases pattern
            match_diseases_selection = True
            age_of_onsets = []

            for n, D in enumerate(diseases):

                patients_with_disease = self.disease2patients[D]

                have_disease = ID in patients_with_disease

                if have_disease == negate_D[n]: 
                    match_diseases_selection = False
                    break

                if have_disease:
                    age_of_onset = patients_with_disease[ID]["age_of_onset"]
                else:
                    age_of_onset = None

                age_of_onsets.append(age_of_onset)

            if not match_diseases_selection: continue

            # Passed all filters. Add record to returned data.
            EMR_data[ID] = copy.deepcopy(self.non_disease_data[ID])

            EMR_data[ID]["age_of_onsets"] = age_of_onsets

        return EMR_data
