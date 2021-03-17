import pandas as pd
import datetime as dt
import os
import glob
from sklearn.cluster import AffinityPropagation

#Read csv with sets and  configs
configs_info=pd.read_csv("../configs/cluster_config.txt", header = None, sep= "=", names=['Desc', 'Value'])
df_opt = pd.read_csv("../data/optimization/StarDust_v1.61_mod_ts_EURCHF_60.csv", sep = ';').set_index('#')

# Parametros de entrada
metrics_list = ['METRIC', 'PROFIT', 'TOTAL TRADES', 'PROFIT FACTOR', 'EXPECTED PAYOFF','EQUITY DD MAX', 'EFICIENCIA CURVA',
                'TRADES PER MONTH', 'PROFIT % PER YEAR', 'CA', 'DW%', 'RELACION 24 MESES', 'RELACION 12 MESES',
                'RELACION 6 MESES', 'MEDIA CO', 'MAXIMO CO']

date_start = configs_info[configs_info.Desc=='date_start'].Value.iloc[0]
date_end = configs_info[configs_info.Desc=='date_end'].Value.iloc[0]
anual_return = float(configs_info[configs_info.Desc=='anual_return'].Value.iloc[0])
max_dd_allow = float(configs_info[configs_info.Desc=='max_dd_allow'].Value.iloc[0])
min_trades_month = float(configs_info[configs_info.Desc=='min_trades_month'].Value.iloc[0])

# Caculating filters
date_start = dt.datetime.strptime(date_start, '%Y-%m-%d')
date_end = dt.datetime.strptime(date_end, '%Y-%m-%d')
total_months = (date_end.year - date_start.year) * 12 + (date_end.month - date_start.month) +1
np_dd_ratio = round((anual_return/12*total_months)/max_dd_allow,2)
total_trades = round(min_trades_month*total_months,0)

print('\nOriginal rows: {}'.format(len(df_opt)))

# Delete repeated params
df_param = df_opt.drop(metrics_list, axis = 1)
df_param.drop_duplicates(inplace = True)
df_opt = df_opt.loc[df_param.index]
df_opt.reset_index(drop = True, inplace = True)

# Cleaning opt
df_metrics = df_opt[metrics_list]
df_metrics = df_metrics[df_metrics['PROFIT FACTOR'] >= 1.3]
df_metrics = df_metrics[df_metrics['RELACION 6 MESES'] >= 0.0]
df_metrics = df_metrics[df_metrics['RELACION 12 MESES'] >= 0.25]
df_metrics = df_metrics[df_metrics['RELACION 24 MESES'] >= 0.5]
df_metrics = df_metrics[(df_metrics['PROFIT'] /  df_metrics['EQUITY DD MAX']) >= np_dd_ratio]
df_metrics = df_metrics[df_metrics['TOTAL TRADES'] >= total_trades]

df_opt = df_opt.loc[df_metrics.index]
df_opt.reset_index(drop = True, inplace = True)

if len(df_opt)<=100:
    print("\n Not enough set to make cluster")
 else:

    #Make clustering
    print("\nMaking clustering...")
    df_param = df_opt.drop(metrics_list, axis = 1)

    # Fit AFfinity Propagation with Scikit
    afprop = AffinityPropagation(max_iter=500, random_state=5, )
    af = afprop.fit(df_param)

    centers = af.cluster_centers_indices_

    df_opt = df_opt.iloc[centers]

print('\nTotal rows: {}'.format(len(df_opt)))

df_opt.to_csv("../data/optimization/StarDust_v1.61_mod_ts_EURCHF_60_updated.csv")