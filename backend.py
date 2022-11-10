import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import streamlit as st


# @st.cache
def get_df(df_original,diff_spliter = 1800):
    """ 
    Выполняет предобработку данных.

    Parameters
    ----------
    df_original : DataFrame
        Первичный df_original.
        
    diff_spliter : int
        Длительность сессии при бездействии в секундах.
    
    Returns
    -------
    frame_lag : DataFrame
        Залинкованный вида [...,site_start,site_end,...].
    """
    frame = df_original
    frame.columns = [col.lower() for col in frame.columns]
    frame.url = frame.url.apply(lambda x: x.lower())
    frame = frame.drop('id_stat',axis = 1).drop_duplicates()
    frame.time_stamp = pd.to_datetime(frame.time_stamp)
    frame = frame.sort_values(by = 'time_stamp')
    frame = frame.groupby(['clientcode','time_stamp','useragent']).first().reset_index()
    frame_lag = pd.concat([frame.shift(periods = 1).rename(columns = {'url':'url_start','time_stamp':'time_stamp_start','useragent':'useragent_start'}),frame[['url','time_stamp','useragent','clientcode']].rename(columns = {'url':'url_end','time_stamp':'time_stamp_end','useragent':'useragent_end','clientcode':'clientcode_end'})],axis = 1).dropna()
    frame_lag['diff'] = (frame_lag.time_stamp_end - frame_lag.time_stamp_start).dt.seconds
    frame_lag.url_start = frame_lag.url_start.apply(lambda x: '/'.join(x.split('/')[2:]).split('.')[0])
    frame_lag.url_end = frame_lag.url_end.apply(lambda x: '/'.join(x.split('/')[2:]).split('.')[0])
    index_to_end = frame_lag[frame_lag['clientcode'] != frame_lag['clientcode_end']].index.tolist()
    columns = {'url_end':'end','time_stamp_end':'0000-00-00 00:00:00', 'useragent_end':'None','clientcode_end':'None', 'diff':-99}
    for col,val in columns.items():
        frame_lag.loc[index_to_end,col] = val
    frame_lag = frame_lag.drop(['clientcode_end'],axis = 1).reset_index(drop = True)
    index_to_del_2 = frame_lag[frame_lag['diff'] == 0].index
    frame_lag = frame_lag.drop(index_to_del_2.values.tolist())
    index_reload = frame_lag[(frame_lag['url_start'] == frame_lag['url_end']) & (frame_lag['diff'] < diff_spliter)].index
    frame_lag['reload'] = 0
    frame_lag.loc[index_reload.tolist(),['reload']] = 1
    frame_lag = frame_lag.reset_index(drop = True)
    ## после завершения процедуры как бы новая сессия
    index_complete_stop = frame_lag[(frame_lag.url_start.apply(lambda x: 'complete' in x)) & (frame_lag['diff'] != -99)].index.tolist()
    # index_complete_stop_first = (np.array(index_complete_stop) + 1).tolist()
    # index_complete_stop = [0]
    # index_complete_stop_first = [1]
    # index_start_point_last = frame_lag[(frame_lag['diff'] >= diff_spliter) | (frame_lag['useragent_start'] != frame_lag['useragent_end'])].index.tolist()
    # index_start_point_last = list(set(index_start_point_last + index_complete_stop))
    # index_start_point_first = (np.array(index_start_point_last) + 1)
    # index_start_point_first = sorted(list(set(index_start_point_first.tolist()  +index_complete_stop_first)))
    # index_start_point_first = index_start_point_first[:-1] + [index_start_point_first[-1]-1]

    index_start_point_last = frame_lag[(frame_lag['diff'] >= diff_spliter) | (frame_lag['useragent_start'] != frame_lag['useragent_end']) | ((frame_lag.url_start.apply(lambda x: 'complete' in x) & (frame_lag['diff'] != -99)))].index.tolist()
    index_start_point_first = (np.array(index_start_point_last) + 1).tolist()
    index_start_point_first = index_start_point_first[:-1] + [index_start_point_first[-1]-1]
    index_to_end = frame_lag[(frame_lag['diff'] >= diff_spliter) | (frame_lag['useragent_start'] != frame_lag['useragent_end'])].index.tolist()
    columns = {'url_end':'end','time_stamp_end':'0000-00-00 00:00:00', 'useragent_end':'None','reload':0,'clientcode_end':'None', 'diff':-99}
    for col,val in columns.items():
        frame_lag.loc[index_to_end,col] = val
    compare = (frame_lag.iloc[index_start_point_first]['clientcode'] == frame_lag.iloc[index_start_point_last]['clientcode'].tolist())
    compare = compare[compare == True].index.tolist()
    frame_lag['start_flag'] = 0
    frame_lag.loc[compare,'start_flag'] = 1
    frame_lag = frame_lag.drop(index_complete_stop).reset_index(drop = True)
    index_to_del_3 = ['start']
    while index_to_del_3:
        index_first_enter = frame_lag.sort_values(by = 'time_stamp_start')[['clientcode','time_stamp_start']].reset_index().groupby(['clientcode']).first()['index'].tolist()
        frame_lag.loc[index_first_enter,['start_flag']] = 1
        index_to_del_3 = frame_lag[(frame_lag['start_flag'] == 1) & ~(frame_lag['url_start'].apply(lambda x: 'main' in x) )].index.tolist()
        frame_lag = frame_lag.drop(index_to_del_3).reset_index(drop = True)
    rank_start = frame_lag.query('start_flag == 1').sort_values(by = 'time_stamp_start')[['clientcode','time_stamp_start']].groupby(['clientcode']).rank(method = 'first',ascending = True)
    rank_start = rank_start.rename(columns = {'time_stamp_start':'start_flag_num'})
    frame_lag = frame_lag.merge(rank_start,left_index = True, right_index = True,how = 'left')
    frame_lag.start_flag_num = frame_lag.start_flag_num.fillna(0)
    rank_process = frame_lag.query('start_flag == 0').sort_values(by = 'time_stamp_start')[['clientcode','time_stamp_start']].groupby(['clientcode']).rank(method = 'first',ascending = True)+1
    rank_process = rank_process.rename(columns = {'time_stamp_start':'process_flag_num'})
    frame_lag = frame_lag.merge(rank_process,left_index = True, right_index = True,how = 'left')
    frame_lag.process_flag_num = frame_lag.process_flag_num.fillna(1)
    index_complete = frame_lag[frame_lag['url_end'].apply(lambda x: 'complete' in x)].index.tolist()
    frame_lag.loc[index_complete,'url_end'] = 'complete'
    def time_filtr(hour):
        if 0 < hour <= 6:
            return 'ночь'
        elif 6 < hour <= 12:
            return 'утро'
        elif 12 < hour < 18:
            return 'день'
        else:
            return 'вечер'
    frame_lag['start_product'] = frame_lag[['url_start','url_end']].apply(lambda x: x[0].split('/')[0] if x[0].split('/')[0] != 'main' else  x[1].split('/')[0],axis = 1)
    frame_lag['day_time'] = frame_lag.time_stamp_start.dt.hour.apply(time_filtr)
    frame_lag['day_of_week'] = frame_lag.time_stamp_start.dt.day_name()
    frame_lag['date_norm'] = frame_lag.time_stamp_start.dt.normalize()
    frame_lag['user_agent_filtr'] = frame_lag.useragent_start.apply(lambda x: x.split('(')[1].split()[0])
    return frame_lag

# @st.cache
def get_metrics(df): 
    """ 
    Высчитывает метрики для залинкованного DataFrame  вида [...,url_start,url_end,...].

    Parameters
    ----------
    df : DataFrame
        залинкованный DataFrame  вида [...,url_start,site_end,...].
    
    Returns
    -------
    (metrics_node, metrics_relation)
    
        metrics_node : 
            DataFrame c метриками для каждой url-вершины (url-node).

        metrics_node : 
            DataFrame c метриками для каждой пары url_start - url_end.
    """

    df['time_stamp_start'] = pd.to_datetime(df.time_stamp_start)
    df['time_stamp_end']   = pd.to_datetime(df.time_stamp_end)


    metrics_relation = df.groupby(['url_start','url_end']).agg({'diff':['mean','max','min','sum','count','std'],'reload':['mean'], 'reload':['mean'], 'process_flag_num':['mean','max','min','std']})
    metrics_relation.columns = [ '_'.join(col) for col in metrics_relation.columns]
    metrics_relation = metrics_relation.rename(columns={'diff_count':'qty_rel_all','process_flag_num_mean':'process_order_mean','reload_mean':'is_reload'})
    metrics_relation = metrics_relation.reset_index()

    qty_all = df.groupby('url_start').count().rename(columns={'clientcode':'qty_all'})[['qty_all']]
    qty_unique_user = df[['clientcode','url_start']].drop_duplicates().groupby('url_start').count().rename(columns={'clientcode':'qty_unique_user'})
    qty_in = metrics_relation.reset_index().groupby('url_start').sum().rename(columns = {'qty_rel_all':'qty'})
    qty_in.columns = qty_in.columns+'_in'
    qty_out = metrics_relation.reset_index().groupby('url_end').sum().rename(columns = {'qty_rel_all':'qty'})
    qty_out.columns = qty_out.columns+'_out'

    metrics_node = pd.merge(qty_all,qty_unique_user,left_index=True, right_index=True, how = 'left')
    metrics_node = pd.merge(metrics_node,qty_in,left_index=True, right_index=True, how = 'left')
    metrics_node = pd.merge(metrics_node,qty_out,left_index=True, right_index=True, how = 'left')
    metrics_node['qty_in_out_ratio'] = metrics_node.qty_in/metrics_node.qty_out
    metrics_node = metrics_node.reset_index()

    return metrics_node, metrics_relation

@st.cache
def get_metrics_by_user(df):
    df_ = df.copy()

    # df_['journey_id'] = df_['start_flag'].cumsum()
    metrics_relation_by_user = df_.groupby(['clientcode','url_start','url_end']).agg({'diff':['mean','max','min','sum','std','count'],'reload':['mean'], 'process_flag_num':['mean','max','min','std']})
    metrics_relation_by_user.columns = [ '_'.join(col) for col in metrics_relation_by_user.columns]
    metrics_relation_by_user = metrics_relation_by_user.rename(columns={'diff_count':'qty_rel_all','process_flag_num_mean':'process_order_mean','reload_mean':'is_reload'})
    return metrics_relation_by_user

@st.cache
def super_function(metrics_relation_by_user,max_users = 10): 
    """ 
    Строит Sankey-диаграмму со значениям metric для n случайных пользовотелей из залинкованного DataFrame.

    Parameters
    ----------
    metrics_relation_by_user : DataFrame
        Залинкованный аггрегированный DataFrame вида [...,url_start,site_end,...]
        
    max_user_rows : int
        Кол-во пользователей для которых строиться Sankey-диаграмма.

    metric : str
        Значения по которым сториться Sankey-диаграмма. Возможно следующее
        ['Количество переходов с предыдущей страницы на следующую',
          'Количество уникальных пользователей',
          'Среднее время посещения страницы',
          'Максимальное время посещения страницы',
          'Минимальное время посещения страницы',
          'Общее время посещения страницы']


    Returns
    -------
    fig : PlotlyFigure 
        Sankey-диаграмма со значениям metric для n случайных пользовотелей из залинкованного DataFrame.
    """

    
    a = metrics_relation_by_user.reset_index()

    a = a[a.clientcode.isin(list(set(a.clientcode))[:max_users])]

    df_random = a.drop(columns=['clientcode'])
    df_random = df_random.groupby(['url_start','url_end'])['qty_rel_all'].sum().reset_index()


    df_random = df_random.copy().reset_index(drop=True)
    df_random['to_drop'] = True
    for i in range(len(df_random)-1):
        if df_random.loc[i,'qty_rel_all'] == df_random.loc[i+1,'qty_rel_all']:
            df_random.loc[i+1,'url_start'] = df_random.loc[i+1,'url_start'] 
            df_random.loc[i,'to_drop'] = np.nan
    df_random = df_random.dropna().drop(columns=['to_drop'])

    keys = list(set(df_random.url_start).union(set(df_random.url_end)))
    d = dict(zip(keys,range(len(keys))))
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = list(d),
            color = "blue"
        ),
        link = dict(
            source = df_random.url_start.replace(d),
            target = df_random.url_end.replace(d),
            value = df_random.qty_rel_all
        ))])
    return fig

@st.cache
def get_problem_nodes(frame_lag,start = '2022-09-21',base = '2022-09-20',end = '2022-09-23',today = '2022-09-24',count_features = 1):
    metrics,_ = get_metrics(frame_lag[frame_lag['date_norm'] == base])
    metrics = metrics.set_index('url_start')
    date_range =  pd.date_range(start = start,end = end)
    for i in date_range:
        var3,var4 = get_metrics(frame_lag[frame_lag['date_norm'] == i])
        var3 = var3.set_index('url_start')
        metrics+=var3
    metrics /= (len(date_range) + 1)
    last_metrics,_ = get_metrics(frame_lag[frame_lag['date_norm'] == today])
    common_index = last_metrics.reset_index().merge(metrics.reset_index(),on = 'url_start').url_start.values.tolist()
    metrics = metrics.loc[common_index,:]
    last_metrics = last_metrics.set_index('url_start')
    last_metrics = last_metrics.loc[common_index,:]
    res = ~(last_metrics.iloc[:,:count_features]*0.1/metrics.iloc[:,:count_features] < 0.95) * 1
    res = (res/res)
    total = []
    for col in res.columns:
        total += list(product(res[res[col] == 1].index.tolist(),[col]))
    return total


# FROM LIZA WITH LOVE

def data_prep(data):
    id_lists = []
    id_list = []
    data['start_flag_'] = data['start_flag'].shift(periods=-1)
    data.loc[data['start_flag_'] == 1, 'url_start']  =  data['url_start'] + '>' +  data['url_end']
    url_start_ = list(data['url_start'])
    all_paths = []
    path = ''
    for id, p in enumerate(url_start_):
        if '>' not in p:
            path += p
            id_list.append(id)
            path += '>'
        else:
            path += p
            id_list.append(id)
            id_lists.append(id_list)
            all_paths.append(path)
            id_list = []
            path = ''
    return all_paths, id_lists
    

def find_popular_paths(result_paths_df, target, top=10):
    target = 'depositcommon/create'
    result_paths_df['target'] = result_paths_df[0].apply(lambda x: f'{target}' in x)
    result_paths_df['target'] = result_paths_df['target'].astype(int)
    result_paths_df['path_without_complete'] = result_paths_df[0].apply(lambda x: x[:x.find(f'>{target}')])
    paths_probabilities = []
    indexes = []

    for unique_path in result_paths_df['path_without_complete'][result_paths_df['target'] == 1].unique():
        true_path = result_paths_df[(result_paths_df['path_without_complete'] == unique_path) & (result_paths_df['target'] == 1)].shape[0]
        indexes.append(result_paths_df[(result_paths_df['path_without_complete'] == unique_path) & (result_paths_df['target'] == 1)].index)
        paths_probabilities.append([unique_path, true_path])
    paths_probabilities = pd.DataFrame(paths_probabilities)
    paths_probabilities[1] = paths_probabilities[1] / paths_probabilities[1].sum()
    paths_probabilities = paths_probabilities.rename(columns={0:'path'})
    paths_probabilities = paths_probabilities.rename(columns={1:'probability'})
    paths_probabilities = paths_probabilities[['path', 'probability']].reset_index(drop=True)
    paths_probabilities['indexes'] = indexes
    paths_probabilities = paths_probabilities.sort_values(by='probability')[-top:]

    return paths_probabilities[['path', 'probability']], list(paths_probabilities['indexes'])


def get_popular_data(data, id_list, indexes):
    return data.loc[list(np.concatenate(np.array(id_list)[np.concatenate(np.array(indexes))]))]

def draw_sankey_popular(data, target, top=5):
    data_, id_list = data_prep(data)
    data_ = pd.DataFrame(data_)
    b, indexes = find_popular_paths(data_, target, top=top)
    new_data = get_popular_data(data, id_list, indexes)
    data_metric = get_metrics_by_user(new_data)
    return super_function(data_metric)
    

