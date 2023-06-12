import pandas as pd
import numpy as np
import os

import src.helper as h
from src.helper import split
from datetime import date

class Preprocessing:
    def __init__(self, data_path):
        os.chdir(data_path + '\\0_raw_questionnaire\\')
        self.data_path = data_path
        mrvr_eng = pd.read_csv(r"mrvr_english.csv", sep=';')
        mrvr_ger = pd.read_csv(r"mrvr_german.csv", sep=';', encoding="ISO-8859-1")

        mrvr_eng['exp_language'] = ['engl'] * len(mrvr_eng)
        mrvr_ger['exp_language'] = ['ger'] * len(mrvr_ger)

        # Unit Test
        print('Checking if columns correspond to each other: ')
        print(mrvr_eng.columns.equals(mrvr_ger.columns))

        # Merge Datasets
        merged = [mrvr_eng, mrvr_ger]
        self.mrvr_quest = pd.concat(merged)

        # Remove unnecessary variables
        self.mrvr_quest = self.mrvr_quest.drop(["SERIAL", "REF", "MAILSENT", "MODE", "QUESTNNR"], axis=1)

        # Drop test rows
        self.mrvr_quest = self.mrvr_quest.dropna(subset=["ID01_01"])
        self.mrvr_quest = self.mrvr_quest[self.mrvr_quest['ID01_01'] != "Test"]
        self.mrvr_quest = self.mrvr_quest[self.mrvr_quest['ID01_01'] != "test"]

        exp_sub = self.mrvr_quest['ID01_01'].unique().tolist()

        # Delete double ID rows based on missing values in rows
        df = self.mrvr_quest[self.mrvr_quest['ID01_01'].isin(exp_sub)]
        df = df.reset_index(drop=True)
        drop_lst = list()
        for i in exp_sub:
            index = np.where(df['ID01_01'] == i)
            if len(index[0]) > 1:
                ind = np.array(index[0])
                print(ind)
                print(df['ID01_01'].loc[ind[0]])
                NA_fst = df.loc[ind[0]].isna().sum().sum()
                print("Number of NaNs values in first given index:", df.loc[ind[0]].isna().sum().sum())
                print(df['ID01_01'].loc[ind[1]])
                NA_snd = df.loc[ind[1]].isna().sum().sum()
                print("Number of NaNs values in second given index:", df.loc[ind[1]].isna().sum().sum())

                if NA_fst > NA_snd:
                    drop_lst.append(ind[0])
                else:
                    drop_lst.append(ind[1])

        # dataframe selecting only valid rows
        self.mrvr_quest = df[~df.index.isin(drop_lst)]

        # drop rows with more than half NaNs
        self.mrvr_quest.dropna(thresh = df.shape[1]/2, axis = 0, inplace = True)

        # convert ID into int
        self.mrvr_quest.loc[:,'ID01_01'] = self.mrvr_quest['ID01_01'].astype(int).tolist()

        print('The length of the cleaned dataset is ', len(self.mrvr_quest))
        print('Unique IDs: ', self.mrvr_quest['ID01_01'].unique())
        print('Number of unique IDs: ', len(self.mrvr_quest['ID01_01'].unique()))

        print('Shape of dataset: ')
        print(self.mrvr_quest.shape)

    def get_dataframe(self):
        return self.mrvr_quest

    def refactor_and_select_columns(self):
        column_lst = ['ID','gender', 'age', 'language','handedness','education','VR_exp','MR_exp','visual_aid',
                      'eye_color','study_field']
        mapping = {'SD01':'gender','SD02_01': 'age', 'SD06': 'language', 'SD03': 'handedness', 'SD04': 'education',
                   'SD08': 'VR_exp', 'SD09': 'MR_exp', 'SD11': 'visual_aid', 'SD12': 'eye_color','SD05_01':'study_field',
                   'ID01_01':'ID'}
        self.mrvr_quest.rename(columns = mapping, inplace=True)

        relabel = ["gender", "handedness", "visual_aid", "eye_color", "education"]
        for col in self.mrvr_quest.columns:
            if col == relabel[0]:
                self.mrvr_quest[col].replace({1.0: "female", 2.0: "male", 3.0: "other"}, inplace=True)
            elif col == relabel[1]:
                self.mrvr_quest[col].replace({1.0: "left", 2.0: "right", 3.0: "dont know"}, inplace=True)
            elif col == relabel[2]:
                self.mrvr_quest[col].replace({1.0: "no", 2.0: "glasses", 3.0: "contact lenses"}, inplace=True)
            elif col == relabel[3]:
                self.mrvr_quest[col].replace({1.0: "brown", 2.0: "blue", 3.0: "green", 4.0: "dont know"}, inplace=True)
            elif col == relabel[4]:
                self.mrvr_quest[col].replace({1.0: "none", 2.0: "elementary school", 3.0: "secondary school",
                                    4.0: "high school", 5.0: "bachelor", 6.0: "master", 7.0: "PhD", 8: "other"},
                                   inplace=True)

        mapping = create_science_variable()
        self.mrvr_quest['science?'] = self.mrvr_quest['study_field'].replace(mapping)
        self.mrvr_quest['science?'] = self.mrvr_quest['science?'].replace({True: 1, False: 0})

        column_lst = column_lst + ['science?']

        #SIMS
        sims_lst = list()
        for i in range(1, 10):
            sims_lst.append('MU01_0' + str(i))
        for j in range(10, 17):
            sims_lst.append('MU01_' + str(j))

        sims_new = ['SIMS1_IM', 'SIMS2_IR', 'SIMS3_ER', 'SIMS4_A', 'SIMS5_IM', 'SIMS6_IR', 'SIMS7_ER', 'SIMS8_A',
                    'SIMS9_IM','SIMS10_IR', 'SIMS11_ER', 'SIMS12_A', 'SIMS13_IM', 'SIMS14_IR', 'SIMS15_ER', 'SIMS16_A']

        sims_res = h.create_dict_from_lists(sims_lst, sims_new.copy())
        self.mrvr_quest.rename(columns=sims_res, inplace=True)
        column_lst = column_lst + sims_new

        df_sims = self.mrvr_quest[sims_new]
        sp = np.array([s.split("_") for s in df_sims.columns])

        # Intrinsic motivation
        im = np.where(sp[:, 1] == 'IM')[0] +1
        df_im = df_sims.iloc[:, im]
        sum_score = df_im.sum(axis=1).tolist()
        self.mrvr_quest['SIMS_IM_sum'] = sum_score

        # Identified regulation
        ir = np.where(sp[:, 1] == 'IR')[0] + 1
        df_ir = df_sims.iloc[:, ir]
        sum_score = df_ir.sum(axis=1).tolist()
        self.mrvr_quest['SIMS_IR_sum'] = sum_score

        # External regulation
        er = np.where(sp[:, 1] == 'ER')[0] + 1
        df_er = df_sims.iloc[:, er]
        sum_score = df_er.sum(axis=1).tolist()
        self.mrvr_quest['SIMS_ER_sum'] = sum_score

        # External regulation
        a = np.where(sp[:, 1] == 'A')[0] + 1
        df_a = df_sims.iloc[:, er]
        sum_score = df_a.sum(axis=1).tolist()
        self.mrvr_quest['SIMS_A_sum'] = sum_score


        sims_sum_lst = ['SIMS_IM_sum','SIMS_IR_sum','SIMS_ER_sum','SIMS_A_sum']
        column_lst = column_lst + sims_sum_lst

        # Willingness to exert effort
        W_lst = list()
        for i in range(1, 9):
            W_lst.append('MU02_0' + str(i))

        W_new = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']

        I_res = h.create_dict_from_lists(W_lst, W_new.copy())
        self.mrvr_quest.rename(columns=I_res, inplace=True)
        column_lst = column_lst + W_new

        df_W = self.mrvr_quest[W_new]
        sum_score = df_W.sum(axis=1).tolist()
        self.mrvr_quest['Willingness_sum'] = sum_score

        column_lst = column_lst + ['Willingness_sum']

        # Interest
        I_lst = list()
        for i in range(1, 5):
            I_lst.append('IN01_0' + str(i))

        I_new = ['I1', 'I2', 'I3', 'I4']

        I_res = h.create_dict_from_lists(I_lst, I_new.copy())
        self.mrvr_quest.rename(columns=I_res, inplace=True)
        column_lst = column_lst + I_new

        df_I = self.mrvr_quest[I_new]
        sum_score = df_I.sum(axis=1).tolist()
        self.mrvr_quest['Interest_sum'] = sum_score

        column_lst = column_lst + ['Interest_sum']

        # BFI2
        bfi_lst = list()

        for b in range(1, 10):
            bfi_lst.append('PS06' + '_0' + str(b))
        for c in range(10, 21):
            bfi_lst.append('PS06' + '_' + str(c))
        for d in range(1, 10):
            bfi_lst.append('PS07' + '_0' + str(d))
        for e in range(10, 21):
            bfi_lst.append('PS07' + '_' + str(e))
        for f in range(1, 10):
            bfi_lst.append('PS08' + '_0' + str(f))
        for g in range(10, 21):
            bfi_lst.append('PS08' + '_' + str(g))

        bfi_new = list()
        for i in range(1,61):
            bfi_new.append("BFI_" + str(i))

        bfi_res = h.create_dict_from_lists(bfi_lst, bfi_new.copy())
        self.mrvr_quest.rename(columns=bfi_res, inplace=True)
        column_lst = column_lst + bfi_new

        rev_lst = [11, 16, 26, 31, 36, 51, 12, 17, 22, 37, 42, 47, 3, 8, 23, 28, 48, 58, 4, 9, 24, 29, 44,
                   49, 5, 25, 30, 45, 50, 55]
        rev_uniq = ["BFI_" + str(s) for s in rev_lst]

        # create reversed variables
        bfi_r = list()
        for i in rev_uniq:
            for col in bfi_new:
                if i == col:
                    self.mrvr_quest[col + 'R'] = self.mrvr_quest[col].replace({1.0: 5.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0})
                    bfi_r.append(col + 'R')
        column_lst = column_lst + bfi_r

        # Concentration
        C_lst = list()
        for i in range(1, 5):
            C_lst.append('MC01_0' + str(i))
        for j in range(1, 2):
            C_lst.append('MC02_0' + str(j))

        C_new = ['C1', 'C2', 'C3', 'C4']
        C_res = h.create_dict_from_lists(C_lst, C_new.copy())
        self.mrvr_quest.rename(columns=C_res, inplace=True)

        self.mrvr_quest['C4R'] = self.mrvr_quest['C4'].replace({1.0: 4.0, 2.0: 3.0, 3.0: 2.0, 4.0: 1.0})

        df_C = self.mrvr_quest[['C1', 'C2', 'C3', 'C4R']]
        sum_score = df_C.sum(axis=1).tolist()
        self.mrvr_quest['Concentration_sum'] = sum_score

        column_lst = column_lst + ['C4R','Concentration_sum']

        # Cognitive load
        self.mrvr_quest.rename(columns={'MC02_01':'CL'}, inplace=True)
        column_lst = column_lst + ['CL']

        self.mrvr_quest = self.mrvr_quest[column_lst]

    def save_dataframe(self):
        os.chdir(self.data_path + "1_preprocessed_questionnaire\\")
        today = date.today()
        self.mrvr_quest.to_csv('{}_preprocessed_questionnaire.csv'.format(today),index=False)
        print('Dataframe is saved as {}_preprocessed_questionnaire.csv'.format(today))
        print('Save location: data\\1_preprocessed_questionnaire\\')


def create_science_variable():
    mapping = {'Lehramt englisch und Geschichte ': False, 'Rechtswissenschaft/Jura': False,
               'Philosophy of Science (Hauptfach), Biology (Nebenfach)': True,
               'Psychologie': False, 'Psychologie ': False,
               'Sozialp�dagogik mit allgemeinbildendem Zweitfach (Deutsch) auf h�heres Lehramt': False,
               'Englisch Linguistik': False, 'Mathematical physics': True, 'Geoscience': True,
               'Molekular medicine': True, 'Physics': True, 'Biologie B.Sc': True,
               'Master of Education: Geographie, Naturwissenschaft und Technik ': True,
               'Bachelor of Education (Spanisch und Englisch)': False, 'Medieninformatik ': True,
               'B.Sc. International Economics ': False, 'Jura': False, 'Erziehungswissenschaft ': False,
               'Molekulare Medizin': True, 'Kognitionswissenschaft': True, 'Erziehungswissenschaft': True,
               'Englisch & Philosophie B.Ed., Sportwissenschaft Gesundheitsf�rderung B.Sc.': True,
               'Rechtswissenschaft ': False, 'Mathematik B.Sc.': True, 'Geowissenschaften ': True,
               'Humanmedizin': True, 'International Business Administration ': False,
               'International Business Administration': False,
               'Bioinformatik M.Sc': True, 'Medienwissenschaften (HF)/Allgemeine Rhetorik (NF)': False,
               'Nano-Science': True, 'Koreanistik': False,
               'Empirische Bildungsforschung und p�dagogische Psychologir': False,
               'Evangelische Theologie': False, 'Englisch, Spanisch, Erziehungswissenschaft': False,
               'Physik, Informatik': True,
               'Humanmedizin ': True, 'Evolution und Ökologie': True, 'Sportwissenschaft': False,
               'Informatik': True, 'Sinologie ': False, 'Pharmazie':True,
               'Geographie': True, 'Geschichtswissenschaft ': False, 'Kunstgeschichte ': False,
               'Empirische Bildungsforschung und Pädagogische Psychologie': False, 'Englisch, Spanisch ': False,
               'Mathematical Physics': True, 'Mathematics': True, 'Rechtswissenschaften': False,
               'Neural Information Processing': True,
               'Archaeology': False, 'Anglistik, Empirische Kulturwissenschaft': False,
               'Klassische Archäologie HF und Sozial und kulturanthropologie NF': False,
               'Mathematik, Chemie': True, "Rechtswissenschaften ": False, "Soziologie": False,
               "Bioinformatik": True, 'Economics and Business Administration': False,
               'Erwachsenenbildung / Weiterbildung':False, 'Englisch und Ev. Theologie / Lehramt':False,
               'Erziehungswissenschaften mit Sozialpädagogik/Erwachsenenbildung B.A':False,
               'Bachelor of Education Chemie / Italienisch / Spanisch':False, 'B.Ed. Philosophie und Englisch':False,
               'Gymnasiales Lehramz (Deutsch, Englisch)':False, 'HF Geschichtswissenschaft / NF Germanistik':False,
               'International Economics':False, 'Politik':False,
               'Englisch & Philosophie B.Ed., Sportwissenschaft Gesundheitsförderung B.Sc.':False,
               'Sozialpädagogik mit allgemeinbildendem Zweitfach (Deutsch) auf höheres Lehramt':False,
               'Physik':True}

    return mapping

