from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from google.colab import widgets
from google.colab import output
import math


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


def complete_covid_data(covid_data, *, last_date=True, selection=None, selection_total_computation=None, do_print=True):
    last_date_ = covid_data['date'].values[-1]  # Oui oui, on pourrait mieux faire, ailleurs aussi d'ailleurs

    if do_print:
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


def load_data_gs(_url, _config, *, last_date=True, worksheet_name='covid_FR', is_covid_data=False, do_print=True):    
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
        data_, last_date_ = complete_covid_data(data_, last_date=last_date, do_print=do_print)

    return data_, last_date_


def load_data(_config, *, last_date=True, consolidation=False, is_covid_data=False, is_hospital_data=False, do_agregate=True, do_print=True):

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
            _config['url_id'], {}, last_date=last_date, worksheet_name=_config['worksheet_id'], is_covid_data=is_covid_data, do_print=do_print
        )

        _id_departement_mapping = _id_departement_mapping[_config['selection_id']]

        hospital_id = _config['id']

        _id_departement_mapping.set_index(hospital_id, drop=True, inplace=True)
        finess_departements_mapping_ = _id_departement_mapping.to_dict()['dep'].copy()

        finess_name_mapping_ = _id_departement_mapping.to_dict()['rs'].copy()

        return finess_departements_mapping_, finess_name_mapping_

    covid_FR_ = None

    if _config['type']=='GS':
        if do_print:
            print('retrieving departements data...')
        if is_covid_data:
            covid_departement_AR_last, _last_date = load_data_gs(
                _config['url'], _config['departements'], last_date=last_date, 
                worksheet_name=_config['worksheet'], is_covid_data=True, 
                do_print=do_print
            )
            departement_set = covid_departement_AR_last['maille_nom'].values
            covid_dep_FR_AR = data_selection(covid_departement_AR_last)

        if is_covid_data and consolidation:
            print('retrieving regions data...')
            covid_region_SP_last, _last_date = load_data_gs(
                _config['url'], _config['regions'], last_date=last_date, 
                worksheet_name=_config['worksheet'], is_covid_data=True, 
                do_print=do_print
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
                _config['url'], _config['lits'], last_date=last_date, 
                worksheet_name=_config['worksheet'], is_covid_data=False, 
                do_print=do_print
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
        covid_data, _last_date = complete_covid_data(covid_data, last_date=last_date, do_print=do_print)
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
        
        if do_print:
            print("cas: "+ str(cases))
            print("reanimations: "+ str(rea))
            print("gueris: "+ str(gc))
            print("deces: "+ str(dc))
    elif is_hospital_data:
        lits_rea = (covid_FR_.sum())

        if do_print:
            print('lits de réanimation récupérés : ' + str(lits_rea))

    return covid_FR_


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


def prepare_ASU_data(_data, _label, _day, *, do_plotting=False):
    time_series_ = None

    if do_plotting:
        time_col = [col for col in _data.columns if ' 00:00:00' in col]
        time_col_day_index = time_col.index(_day)
        time_col = time_col[:time_col_day_index+1]
        data_plot = _data.copy()
        data_plot.set_index('Site', drop=True, inplace=True)
        for col in time_col:
            data_plot.loc[:,col] = data_plot[col].apply(lambda x: float(str(x).replace(' ', '')) if str(x) != '' else 0)
        
        time_series_ = data_plot[time_col].transpose()
        time_series_.plot(title=_label, figsize=(20,5));

    data_ = _data[['Site', _day]].copy()
    data_.columns = ['asu', _label]
    data_.loc[:, _label] = data_[_label].apply(lambda x: float(str(x).replace(' ', '')) if str(x) != '' else 0)
    data_.set_index('asu', drop=True, inplace=True)

    return data_, time_series_


def plotting_figure(_ax, _data, *, title=None, legend=True):
    fig_ = _data.plot(ax=_ax, legend=legend)
    _ax.set_ylabel('Lits')
    
    if title is not None:
        _ax.set_title(title)
    
    fig_.axes.spines['top'].set_visible(False)
    fig_.axes.spines['right'].set_visible(False)

    return True


def plot_capacity_vs_covid19(_data, dep_set, nrow, ncols, selection_set, *, figsize=None, hspace=None, wspace=None, suptitle=None):
    if figsize is None:
        _fig, _axs = plt.subplots(nrows=nrow, ncols=ncols)
    else:
        _fig, _axs = plt.subplots(nrows=nrow, ncols=ncols, figsize=figsize)
    
    if suptitle is not None:
        _fig.suptitle(str(suptitle))

    if hspace is not None:
        plt.subplots_adjust(hspace=hspace)
    
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)
    
    i=0
    _legend=True
    for _row in _axs:
        for _col in _row:
            _dep = dep_set[i]
            data_ = _data[_data.index.get_level_values(0) == _dep][selection_set].copy()
            data_ = data_.droplevel(0, axis=0)
            data_.index = [v.replace('2020-', '') for v in data_.index]
            plotting_figure(_col, data_[selection_set], title=_dep, legend=_legend)
            _legend=False
            i+=1
    
    return True


def capa_vs_covid19_plotting(_data, selection, *, figsize=(15,15), hspace=0.4, wspace=0.1, not_plotted=None):

    if not_plotted is None:
        not_plotted = list()

    regions_departements_mapping = pd.read_csv('https://www.data.gouv.fr/en/datasets/r/987227fb-dcb2-429e-96af-8979f97c9c84')
    regions_departements_mapping = regions_departements_mapping[['dep_name', 'region_name']]
    gb_ = regions_departements_mapping.groupby('region_name')
    dep_name_series = gb_['dep_name'].apply(list)
    dep_name_series_mapping = dep_name_series.to_dict()

    dep_name_series_mapping_plot = {k:(v, max(math.ceil(len(v)/2.0), 2)) for k, v in dep_name_series_mapping.items() if k not in not_plotted}
    _d = list(dep_name_series_mapping_plot.keys())

    tb = widgets.TabBar(_d, location='start')

    for i in _d:
        with tb.output_to(i):
            dep_set, rows = dep_name_series_mapping_plot[i]
            try:
                plot_capacity_vs_covid19(_data, dep_set, rows, 2, selection, figsize=figsize, hspace=hspace, wspace=wspace);
            except IndexError:
                pass
    
    return True


def get_LOX_consumption_data(_data, _month_inf, _month_sup, _dep_selection, _top_clients, _client_id_mapping_by_ref, *, do_printing=True):
    data = _data.copy()

    # I only get the useful informations
    data = data[[col for col in data.columns if col!='Invoicing Company Name']]
    data.columns = ['ref', 'date', 'consommation totale', 'accessibilite', 'consommation', 'dep']

    # I make sure that the date are in datetime format and I set it as the index
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y")
    data.set_index(['date'], drop=True, inplace=True)

    # I apply the month selection
    data = data[
                        (data.index.month > _month_inf) & 
                        (data.index.month < _month_sup)
    ]

    # I convert the consumptions into numerical data
    data = data.reset_index()
    data.loc[:,'consommation totale'] = data['consommation totale'].apply(
        lambda x: float(str(x).replace(',', '.'))
    )
    data.loc[:,'consommation'] = data['consommation'].apply(
        lambda x: float(str(x).replace(',', '.'))
    )

    # We use the first two numbers of the city code to assess the departement number
    data.loc[:,'dep'] = data['dep'].apply(lambda x: str(x).zfill(5)[:2]).copy()
    conso_ref_TS_dep = data[data['dep'].isin(_dep_selection)]

    # I sum the consumptions over days and I concatenate the clients refs into lists
    gb_date = conso_ref_TS_dep.groupby('date')
    ref_ = gb_date['ref'].apply(list).reset_index(name='ref')
    conso_totale_ = gb_date['consommation totale'].sum().reset_index(name='consommation totale')
    conso_ = gb_date['consommation'].sum().reset_index(name='consommation')
    ref_ = ref_.merge(conso_totale_)
    ref_ = ref_.merge(conso_)

    # I add the clients numbers per day
    ref_['nombre de clients'] = ref_['ref'].apply(lambda x: len(x)).copy()

    # I set the date column as the index
    ref_.set_index('date', drop=True, inplace=True)

    # I get the overall consumptions
    conso_ref_dep = conso_ref_TS_dep.copy()
    gp_ref = conso_ref_dep.groupby('ref')
    conso_sum_dep = gp_ref[['consommation totale']].sum()
    conso_sum_dep = conso_sum_dep.sort_values(by='consommation totale', ascending=False)

    conso_totale_dep = 1

    stats_ = pd.DataFrame()

    conso_totale_dep = int(conso_sum_dep.sum())
    nombre_clients_dep = len(conso_sum_dep.index.values)

    stats_.loc["Consommation (L)", "Total"] = conso_totale_dep
    stats_.loc["Nombre de clients", "Total"] = nombre_clients_dep

    if do_printing:
        print("Consommation totale : " + str(conso_totale_dep))
        print("Nombre de clients : " + str(nombre_clients_dep))

    if _top_clients is not None:
        conso_sum_dep = conso_sum_dep.loc[conso_sum_dep.index.values[:_top_clients]]
    
    conso_sum_dep.index = [v[2:] for v in conso_sum_dep.index.values]  # The first two character are not present in the LOX customer clients ref
    conso_sum_dep = conso_sum_dep.merge(_client_id_mapping_by_ref, left_index=True, right_index=True)
    conso_sum_dep.loc[:,'capacite (L)'] = conso_sum_dep['capacite (L)'].apply(lambda x: float(str(x).replace(',', '.')))
    conso_sum_dep.loc[:,'capacite de secours'] = conso_sum_dep['capacite de secours'].apply(lambda x: float(str(x).replace(',', '.')))

    gb_id = conso_sum_dep.groupby('id')
    conso_ = gb_id['consommation totale'].sum().reset_index(name='consommation')
    accessibilite_ = gb_id['accessibilite'].apply(list).reset_index(name='accessibilite')
    capa_ = gb_id['capacite (L)'].sum().reset_index(name='capacite (L)')
    capa_secours_ = gb_id['capacite de secours'].sum().reset_index(name='capacite de secours')

    conso_ = conso_.merge(accessibilite_).merge(capa_).merge(capa_secours_)
    conso_ = conso_.sort_values(by='consommation', ascending=False)

    conso_totale_dep = int(conso_['consommation'].sum()) / (conso_totale_dep * 1.0) * 100.0
    conso_totale_dep = round(conso_totale_dep, 1)
    nombre_clients_dep = len(conso_.index.values) / (nombre_clients_dep * 1.0) * 100.0
    nombre_clients_dep = round(nombre_clients_dep, 1)

    stats_.loc["Consommation (L)", "Part identifiée (%)"] = conso_totale_dep
    stats_.loc["Nombre de clients", "Part identifiée (%)"] = nombre_clients_dep

    if do_printing:
        print("Consommation totale identifiée : " + str(conso_totale_dep) + "%")
        print("Nombre de clients identifiés : " + str(nombre_clients_dep) + "%")

    conso_.reset_index(drop=True, inplace=True)

    return ref_, conso_, conso_totale_dep, stats_


def consumption_rolling_plot(_data, _rolling_level, _ratio, *, do_plotting=True):
    ref_plot = _data[['consommation totale', 'consommation', 'nombre de clients']].copy()
    ref_plot['ratio'] = ref_plot['consommation totale'] / ref_plot['consommation']
    ref_plot['consommation totale'] = ref_plot['consommation totale'] / (_ratio)
    
    if do_plotting:
        ref_plot['consommation totale'].rolling(_rolling_level).mean().plot();
        plt.figure()
        ref_plot['nombre de clients'].plot();
        plt.figure()

    return ref_plot


def plotting_region_consumption(_data_mapping, _rolling, _figsize):
    _d = list(_data_mapping.keys())

    tb = widgets.TabBar(_d, location='start')

    for i in _d:
        with tb.output_to(i):
            _asu, _LOX, _LOX_ALSF, test_plot, test_stats = _data_mapping[i]

            d_ = ['enlèvements', 'covid 19', 'market share']
            tb_ = widgets.TabBar(d_, location='top')

            for j in d_:
                with tb_.output_to(j):
                    if j=='enlèvements':
                        try:
                            LOX_rolling_plot = _LOX['consommation totale'].rolling(_rolling).sum()
                            LOX_ALSF_rolling_plot = _LOX_ALSF['consommation totale'].rolling(_rolling).sum()
                            asu_rolling_plot = _asu.rolling(_rolling).sum()
                            asu_rolling_plot_ = list(asu_rolling_plot.columns)
                            asu_rolling_plot_ = ['client LOX médical externes', 'client LOX médical internes'] + asu_rolling_plot_
                            asu_rolling_plot['client LOX médical externes'] = LOX_rolling_plot
                            asu_rolling_plot['client LOX médical internes'] = LOX_ALSF_rolling_plot
                            asu_rolling_plot = asu_rolling_plot[asu_rolling_plot_]
                            asu_rolling_plot[asu_rolling_plot.index.month==3].plot(figsize=_figsize);
                        except IndexError:
                            pass
                    elif j=='covid 19':
                        test_plot.plot(figsize=_figsize)
                    else:
                        print(test_stats)

                        
def plotting_region_consumption_(_data_mapping, _data_dep, _rolling, _figsize):
    _d = list(_data_mapping.keys())

    tb = widgets.TabBar(_d, location='start')

    for i in _d:
        with tb.output_to(i):
            _asu, _LOX, _LOX_ALSF, test_plot, test_stats, dep_set = _data_mapping[i]

            d_ = ['enlèvements', 'covid 19', 'market share']
            tb_ = widgets.TabBar(d_, location='top')

            for j in d_:
                with tb_.output_to(j):
                    if j=='enlèvements':                        
                        try:
                            LOX_rolling_plot = _LOX['consommation totale'].rolling(_rolling).sum()
                            LOX_ALSF_rolling_plot = _LOX_ALSF['consommation totale'].rolling(_rolling).sum()
                            asu_rolling_plot = _asu.rolling(_rolling).sum()
                            asu_rolling_plot_ = list(asu_rolling_plot.columns)
                            asu_rolling_plot_ = ['client LOX médical externes', 'client LOX médical internes'] + asu_rolling_plot_
                            asu_rolling_plot['client LOX médical externes'] = LOX_rolling_plot
                            asu_rolling_plot['client LOX médical internes'] = LOX_ALSF_rolling_plot
                            asu_rolling_plot = asu_rolling_plot[asu_rolling_plot_]
                            asu_rolling_plot[asu_rolling_plot.index.month==3].plot(figsize=_figsize);
                        except IndexError:
                            pass
                    elif j=='covid 19':
                        tb_dep = widgets.TabBar(dep_set, location='bottom')
                        
                        data_plot_region = _data_dep[['capacité (2018)']].copy()
                        data_plot_region = data_plot_region[data_plot_region.index.get_level_values(0).isin(dep_set)]
                        data_plot_region = data_plot_region.groupby(level=[1]).sum()
                        test_plot['capacité (2018)'] = data_plot_region['capacité (2018)']
                        test_plot = test_plot[['en reanimation ou soins intensifs', 'capacité (2018)', 'hospitalises', 'deces']]
                        
                        for d in dep_set:
                            with tb_dep.output_to(d):

                                _fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=_figsize);
                                plotting_figure(_axs[0], test_plot, title=i);
                                data_plot_ = _data_dep.loc[d, :][['reanimation', 'capacité (2018)']]
                                plotting_figure(_axs[1], data_plot_, title=d);
                                plt.show();

                    else:
                        print(test_stats)
