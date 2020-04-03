from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import pandas as pd
import requests
import geopandas as gpd


def get_geometries(_geo_request):

    geo_, code_departement_mapping_ = None, None

    _country = _geo_request.get('country', '')

    _code_mapping = _geo_request.get('code_mapping', False)
    
    if isinstance(_code_mapping, bool):
        code_mapping = _code_mapping
    else:
        code_mapping = False

    if _country.lower() == 'france':
        geojson_france_departements = _geo_request['url_departements']
        geo_departements = gpd.read_file(geojson_france_departements)

        # J'en profite pour récupérer le mapping entre les codes et les noms des départements
        if code_mapping:
            code_departement_mapping = geo_departements.copy()
            code_departement_mapping = code_departement_mapping[['code', 'nom']]
            code_departement_mapping.set_index('code', drop=True, inplace=True)
            code_departement_mapping_ = code_departement_mapping.to_dict()['nom']

        # Je récupère les géométries des régions françaises
        geojson_france_regions = _geo_request['url_regions']
        geo_regions = gpd.read_file(geojson_france_regions)
        geo_regions.columns = ['nom', 'geometry']
        
        if 'correction_nom_regions' in _geo_request:
            geo_regions['nom'] = pd.Series(_geo_request['correction_nom_regions'])

        # J'assemble les géométries
        geo_FR_ = geo_regions.copy()
        geo_FR_ = geo_FR_.append(geo_departements[['nom', 'geometry']], ignore_index=True)
        geo_FR_.set_index('nom', drop=True, inplace=True)
        geo_ = geo_FR_
    else:
        raise 'Sorry but France is the only country available right now'

    return geo_, code_departement_mapping_


def prepare_data_customer(_data, dep_code_mapping=None, mandatory_col=None):
    data = _data.copy()

    if mandatory_col is not None:
        if all([col in data.columns for col in mandatory_col]):
            data = data[mandatory_col]
        else:
            print('These columns are mandatory: ' + str(mandatory_col))

    data['Capa Principale (L)'] = data['Capa Principale (L)'].apply(lambda x: round(float(str(x).replace(',', '.')), 2) if x!='' else 0).copy()

    groupby_departement = data.groupby(['Departement'])
    clients = groupby_departement['Nom Client. 1'].apply(list).reset_index(name='clients')
    ASUs = groupby_departement['Source Defaut'].apply(list).reset_index(name='ASUs')

    capa = groupby_departement['Capa Principale (L)'].sum().reset_index(name='capacite agregee (L)')

    ASUs.loc[:, 'ASUs'] = ASUs['ASUs'].apply(lambda x: set(x)).copy()

    data_ = clients.merge(ASUs)
    data_ = data_.merge(capa)

    groupby_ASUs = data.groupby(['Source Defaut'])
    clients = groupby_ASUs['Nom Client. 1'].apply(list).reset_index(name='clients')
    departements = groupby_ASUs['Departement'].apply(list).reset_index(name='departements')

    capa = groupby_ASUs['Capa Principale (L)'].sum().reset_index(name='capacite agregee (L)')

    departements.loc[:, 'departements'] = departements['departements'].apply(lambda x: set(x)).copy()

    data_ASUs = clients.merge(departements)
    data_ASUs = data_ASUs.merge(capa)

    if dep_code_mapping is not None:
        try:
            data_['nom'] = data_['Departement'].apply(lambda x: dep_code_mapping[str(x).zfill(2)])
            data_.set_index('nom', drop=True, inplace=True)
            data_.drop('Departement', axis=1, inplace=True)
        except KeyError:  # TODO: tobe improved
            pass

    return data_, data_ASUs


def complete_covid_data(covid_data, *, last_date=True, selection=None, selection_total_computation=None):
    last_date_ = covid_data['date'].values[-1]  # Oui oui, on pourrait mieux faire, ailleurs aussi d'ailleurs
    print("last date update: "+ last_date_)
    if last_date:
        covid_data = covid_data[covid_data['date'] == last_date_]

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

    return covid_data, last_date_


def load_data_gs(_url, _config, *, last_date=True, worksheet_name='covid_FR', is_covid_data=False):    
    gc = gspread.authorize(GoogleCredentials.get_application_default())
    wb = gc.open_by_url(_url)
    worksheet = wb.worksheet(worksheet_name)
    _data = worksheet.get_all_values()

    data_df = pd.DataFrame(_data)
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:]

    data_ = data_df.copy()

    for key, value in _config.items():
        if isinstance(value, list):
            data_ = data_[data_[key].isin(value)]
        else:
            data_ = data_[data_[key]==value]
    
    last_date_ = None

    if is_covid_data:
        data_, last_date_ = complete_covid_data(data_, last_date=last_date)

    return data_, last_date_


def load_data(_config, *, last_date=True, consolidation=False, is_covid_data=False, is_hospital_data=False, do_agregate=True):

    auth.authenticate_user()

    def data_selection(_data):
        data_ = None

        if is_covid_data:  # TODO: to be removed

            if last_date:
                selection = ['maille_nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation']
                selection_alt = ['maille_nom', 'cas_confirmes', 'deces', 'gueris', 'hospitalises', 'reanimation']
            else:
                selection = ['maille_nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation', 'date']
                selection_alt = ['maille_nom', 'cas_confirmes', 'deces', 'gueris', 'hospitalises', 'reanimation', 'date']

            try:
                data_ = _data[selection].copy()
            except KeyError:
                data_ = _data[selection_alt].copy()

            if last_date:
                data_.columns = ['nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation']
                data_.set_index('nom', drop=True, inplace=True)
            else:
                data_.columns = ['nom', 'cas', 'deces', 'gueris', 'hospitalises', 'reanimation', 'date']
                data_.set_index(['nom', 'date'], drop=True, inplace=True)
        elif is_hospital_data:
            data_ = _data[_config['selection']].copy()

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

    def get_finess_departements_mapping():
        finess_departements_mapping_ = None

        _id_departement_mapping, _ = load_data_gs(
            _config['url_id'], {}, last_date=last_date, worksheet_name=_config['worksheet_id'], is_covid_data=is_covid_data
        )

        _id_departement_mapping = _id_departement_mapping[_config['selection_id']]

        hospital_id = _config['id']

        _id_departement_mapping.set_index(hospital_id, drop=True, inplace=True)
        finess_departements_mapping_ = _id_departement_mapping.to_dict()['dep'].copy()

        finess_name_mapping_ = _id_departement_mapping.to_dict()['rs'].copy()

        return finess_departements_mapping_, finess_name_mapping_

    covid_FR_ = None

    if _config['type']=='GS':
        print('retrieving departements data...')
        if is_covid_data:
            covid_departement_AR_last, _last_date = load_data_gs(
                _config['url'], _config['departements'], last_date=last_date, worksheet_name=_config['worksheet'], is_covid_data=True
            )
            departement_set = covid_departement_AR_last['maille_nom'].values
            covid_dep_FR_AR = data_selection(covid_departement_AR_last)

        if is_covid_data and consolidation:
            print('retrieving regions data...')
            covid_region_SP_last, _last_date = load_data_gs(
                _config['url'], _config['regions'], last_date=last_date, worksheet_name=_config['worksheet'], is_covid_data=True
            )
            regions_set = covid_region_SP_last['maille_nom'].values

            regions_departements_mapping = get_regions_departements_mapping()
            selection_ = list()
            for key, value in regions_departements_mapping.items():
                if all([d in departement_set for d in value]):
                    selection_ = selection_ + value
                else:
                    selection_.append(key)

            covid_reg_FR_SP = data_selection(covid_region_SP_last)
            covid_FR_ = covid_reg_FR_SP.append(covid_dep_FR_AR).loc[selection_]
        elif is_covid_data:
            covid_FR_ = covid_dep_FR_AR
        
        elif is_hospital_data:
            hospital_departement, _ = load_data_gs(
                _config['url'], _config['lits'], last_date=last_date, worksheet_name=_config['worksheet'], is_covid_data=False
            )

            hospital_dep_FR = data_selection(hospital_departement)

            hospital_id = _config['id']  # Safety improvement to do here?

            id_departement_mapping, id_departement_mapping_name = get_finess_departements_mapping()

            hospital_dep_FR['dep'] = hospital_dep_FR[hospital_id].apply(lambda x: id_departement_mapping[x])
            hospital_dep_FR['rs'] = hospital_dep_FR[hospital_id].apply(lambda x: id_departement_mapping_name[x])

            hospital_dep_FR = hospital_dep_FR[['rs', 'dep', 'LIT']]

            hospital_dep_FR.columns = ['etablissement','dep', 'lits']

            hospital_dep_FR['nom'] = hospital_dep_FR['dep']
            code_departement_mapping = _config['dep_code_mapping']
            hospital_dep_FR.loc[:, 'nom'] = hospital_dep_FR['nom'].apply(lambda x: code_departement_mapping[str(x).zfill(2)] if str(x).zfill(2) in code_departement_mapping else '').copy()
            hospital_dep_FR = hospital_dep_FR[hospital_dep_FR['nom']!='']

            hospital_dep_FR = hospital_dep_FR[['etablissement','nom', 'lits']]

            hospital_dep_FR.loc[:, 'lits'] = hospital_dep_FR['lits'].apply(lambda x: int(x) if x is not '' else 0).copy()

            if do_agregate:
                hospital_dep_FR = hospital_dep_FR.groupby(['nom']).sum()

            covid_FR_ = hospital_dep_FR  # TODO: complete with departements names and proceed to agregation

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
        covid_data, _last_date = complete_covid_data(covid_data, last_date=last_date)
        covid_FR_ = data_selection(covid_data)

    if is_covid_data:
        covid_FR_.loc[:, 'cas'] = covid_FR_['cas'].apply(lambda x: 0 if (x=='' or x==' ') else int(x)).copy()

        if last_date:
            cases = covid_FR_['cas'].sum()
            rea = covid_FR_['reanimation'].sum()
            gc = covid_FR_['gueris'].sum()
            dc = covid_FR_['deces'].sum()
        else:
            print_results = covid_FR_[covid_FR_.index.get_level_values(1) == _last_date]
            cases = print_results['cas'].sum()
            rea = print_results['reanimation'].sum()
            gc = print_results['gueris'].sum()
            dc = print_results['deces'].sum()
        print("cas: "+ str(cases))
        print("reanimations: "+ str(rea))
        print("gueris: "+ str(gc))
        print("deces: "+ str(dc))
    elif is_hospital_data:
        lits_rea = (covid_FR_.sum())
        print('lits de réanimation récupérés : ' + str(lits_rea))

    return covid_FR_
