import pandas as pd
import numpy as np
import os
import glob
import datetime
from datetime import datetime


# Function to split a string into a list of letters
def split(word):
    return [char for char in word]


class Preprocessing:
    def __init__(self, data_path):
        os.chdir(data_path + '/0_raw_questionnaire/')
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

    #TODO ab hier
    def create_study_field(self):
        self.df['study_field'].replace({'Lehramt englisch und Geschichte ': False, 'Rechtswissenschaft/Jura': False,
                                   'Philosophy of Science (Hauptfach), Biology (Nebenfach)': True,
                                   'Psychologie': False, 'Psychologie ': False,
                                   'Sozialp�dagogik mit allgemeinbildendem Zweitfach (Deutsch) auf h�heres Lehramt': False,
                                   'Englisch Linguistik': False, 'Mathematical physics': False, 'Geoscience': True,
                                   'Molekular medicine': True, 'Physics': True, 'Biologie B.Sc': True,
                                   'Master of Education: Geographie, Naturwissenschaft und Technik ': True,
                                   'Bachelor of Education (Spanisch und Englisch)': False, 'Medieninformatik ': True,
                                   'B.Sc. International Economics ': False, 'Jura': False,
                                   'Erziehungswissenschaft ': False,
                                   'Molekulare Medizin': True, 'Kognitionswissenschaft': True,
                                   'Erziehungswissenschaft': True,
                                   'Englisch & Philosophie B.Ed., Sportwissenschaft Gesundheitsf�rderung B.Sc.': True,
                                   'Rechtswissenschaft ': False, 'Mathematik B.Sc.': True, 'Geowissenschaften ': True,
                                   'Humanmedizin': True, 'International Business Administration ': False,
                                   'International Business Administration': False,
                                   'Bioinformatik M.Sc': True,
                                   'Medienwissenschaften (HF)/Allgemeine Rhetorik (NF)': False,
                                   'Nano-Science': True, 'Koreanistik': False,
                                   'Empirische Bildungsforschung und p�dagogische Psychologir': False,
                                   'Evangelische Theologie': False, 'Englisch, Spanisch, Erziehungswissenschaft': False,
                                   'Physik, Informatik': False,
                                   'Humanmedizin ': True, 'Evolution und �kologie': True, 'Sportwissenschaft': False,
                                   'Informatik': True, 'Sinologie ': False,
                                   'Geographie': True, 'Geschichtswissenschaft ': False, 'Kunstgeschichte ': False,
                                   'Empirische Bildungsforschung und p�dagogische Psychologie ': False,
                                   'Englisch, Spanisch ': False,
                                   'Mathematical Physics': True, 'Mathematics': True, 'Rechtswissenschaften': False,
                                   'Neural Information Processing': True,
                                   'Archaeology': False, 'Anglistik, Empirische Kulturwissenschaft': False,
                                   'Klassische Arch�ologie HF und Sozial und kulturanthropologie NF': False,
                                   'Mathematik, Chemie': True, "Rechtswissenschaften ": False, "Soziologie": False,
                                   "Bioinformatik": True, 'Economics and Business Administration': False},
                                  inplace=True)
        mapping = {'Lehramt englisch und Geschichte ': False, 'Rechtswissenschaft/Jura': False,
                   'Philosophy of Science (Hauptfach), Biology (Nebenfach)': True,
                   'Psychologie': False, 'Psychologie ': False,
                   'Sozialp�dagogik mit allgemeinbildendem Zweitfach (Deutsch) auf h�heres Lehramt': False,
                   'Englisch Linguistik': False, 'Mathematical physics': False, 'Geoscience': True,
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
                   'Physik, Informatik': False,
                   'Humanmedizin ': True, 'Evolution und �kologie': True, 'Sportwissenschaft': False,
                   'Informatik': True, 'Sinologie ': False,
                   'Geographie': True, 'Geschichtswissenschaft ': False, 'Kunstgeschichte ': False,
                   'Empirische Bildungsforschung und p�dagogische Psychologie ': False, 'Englisch, Spanisch ': False,
                   'Mathematical Physics': True, 'Mathematics': True, 'Rechtswissenschaften': False,
                   'Neural Information Processing': True,
                   'Archaeology': False, 'Anglistik, Empirische Kulturwissenschaft': False,
                   'Klassische Arch�ologie HF und Sozial und kulturanthropologie NF': False,
                   'Mathematik, Chemie': True, "Rechtswissenschaften ": False, "Soziologie": False,
                   "Bioinformatik": True, 'Economics and Business Administration': False}

        self.df['science?'] = self.df['study_field'].replace(mapping)
        self.df['science?'] = self.df['science?'].replace({True: 1, False: 0})
