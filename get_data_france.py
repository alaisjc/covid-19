from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import pandas as pd
import requests

import warnings
warnings.filterwarnings('ignore')

def complete_data(covid_data, *, last_date=True, selection=None, selection_total_computation=None):
    if last_date:
        last_date = covid_data['date'].values[-1]  # Oui oui, on pourrait mieux faire, ailleurs aussi d'ailleurs
        print("last date update: "+ last_date)
        covid_data = covid_data[covid_data['date'] == last_date]

    if 'cas_confirmes' in covid_data.columns:
        covid_data.loc[:, 'cas_confirmes'] = covid_data['cas_confirmes'].apply(lambda x: 0 if (x=='' or x==' ') else int(x)).copy()  # Ici, par exemple ?
    else:
        covid_data['cas_confirmes'] = 0

    if selection is None:
        selection = ['cas_confirmes', 'deces', 'hospitalises', 'gueris', 'reanimation']

    if selection_total_computation is None:
        selection_total_computation = ['deces', 'hospitalises', 'gueris']

    for col in selection:
        covid_data.loc[:, col] = covid_data[col].apply(lambda x: 0 if (x=='' or x==' ') else int(x)).copy()

    covid_data["cas_confirmes_"] = covid_data[selection_total_computation].sum(axis=1)
    covid_data["cas"] = covid_data[['cas_confirmes_', 'cas_confirmes']].max(axis=1)

    return covid_data

def load_data_gs(_url, _config, *, last_date=True, worksheet_name='covid_FR'):

    _granularite = _config['granularite']
    _source_type = _config['source_type']

    auth.authenticate_user()
    gc = gspread.authorize(GoogleCredentials.get_application_default())
    wb = gc.open_by_url(_url)
    worksheet = wb.worksheet(worksheet_name)
    covid_data = worksheet.get_all_values()

    covid_df = pd.DataFrame(covid_data)
    covid_df.columns = covid_df.iloc[0]
    covid_df = covid_df.iloc[1:]

    covid_data = covid_df[covid_df['granularite'] == _granularite]
    covid_data = covid_data[covid_data['source_type'] == _source_type]

    covid_data = complete_data(covid_data, last_date=last_date)

    return covid_data


def load_data(_config, *, last_date=True, consolidation=False):

    def data_selection(_data):
        try:
            data_ = _data[['maille_nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation']].copy()
        except KeyError:
            data_ = _data[['maille_nom', 'cas_confirmes', 'deces', 'gueris', 'hospitalises', 'reanimation']].copy()

        data_.columns = ['nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation']
        data_.set_index('nom', drop=True, inplace=True)

        return data_

    def get_regions_departements_mapping():
        regions_departements_set = requests.get(_config['url_regions_departements']).json()
        regions_departements_mapping_ = dict()
        for value in regions_departements_set:
            dep_name = value['dep_name']
            region_name = value['region_name']

            if region_name in regions_departements_mapping_:
                regions_departements_mapping_[region_name].append(dep_name)
            else:
                regions_departements_mapping_[region_name] = [dep_name]
        return regions_departements_mapping_

    if _config['type']=='GS':

        if consolidation:
            print('retrieving regions data...')
            covid_region_SP_last = load_data_gs(
                _config['url'], _config['regions'], last_date=last_date, worksheet_name=_config['worksheet']
            )

            print('retrieving departements data...')
            covid_departement_AR_last = load_data_gs(
                _config['url'], _config['departements'], last_date=last_date, worksheet_name=_config['worksheet']
            )

            regions_set = covid_region_SP_last['maille_nom'].values
            departement_set = covid_departement_AR_last['maille_nom'].values

            regions_departements_mapping = get_regions_departements_mapping()
            selection_ = list()
            for key, value in regions_departements_mapping.items():
                if all([d in departement_set for d in value]):
                    selection_ = selection_ + value
                else:
                    selection_.append(key)

            covid_reg_FR_SP = data_selection(covid_region_SP_last)
            covid_dep_FR_AR = data_selection(covid_departement_AR_last)
            covid_FR_ = covid_reg_FR_SP.append(covid_dep_FR_AR).loc[selection_]
        else:
            print('retrieving departements data...')
            covid_departement = load_data_gs(
                _config['url'], _config['departements'], last_date=last_date, worksheet_name=_config['worksheet']
            )
            covid_FR_ = data_selection(covid_departement)

    elif _config['type']=='datagouv':
        covid_data = pd.read_csv(_config['url'], sep=";", infer_datetime_format=True)
        dep_ = pd.read_csv(_config['url_departements_mapping'], sep=",")
        dep_ = dep_[['num_dep', 'dep_name']]
        dep_.set_index('num_dep', drop=True, inplace=True)
        dep_mapping = dep_.to_dict()['dep_name']

        # We do not consider the sex distinction and we replace the departements numbers by the departements names
        covid_data = covid_data[covid_data.sexe==_config['sexe']][[col for col in covid_data.columns if col!='sexe']]
        covid_data.loc[:, 'dep'] = covid_data.dep.apply(lambda x: dep_mapping.get(x,'')).copy()
        covid_data.columns = ['maille_nom', 'date', 'hospitalises', 'reanimation', 'gueris', 'deces']
        covid_data = complete_data(covid_data, last_date=last_date)
        covid_FR_ = data_selection(covid_data)

    covid_FR_.loc[:, 'cas'] = covid_FR_['cas'].apply(lambda x: 0 if (x=='' or x==' ') else int(x)).copy()
    cases = covid_FR_['cas'].sum()
    rea = covid_FR_['reanimation'].sum()
    gc = covid_FR_['gueris'].sum()
    dc = covid_FR_['deces'].sum()

    print("cas: "+ str(cases))
    print("reanimations: "+ str(rea))
    print("gueris: "+ str(gc))
    print("deces: "+ str(dc))

    return covid_FR_
