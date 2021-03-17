from tools.stat_tools.stat_tools import *
import pandas as pd
import numpy as np
import os
import glob
import sys

path_opt = sys.argv[1]

dd_allow = 1000
mean_loss_allow = 100
max_loss_allow = 200
lots_var_name = 'Lots'

time_frame = path_opt.split("_")[-1]

if time_frame == "M1":
    time_frame_int = '1'
elif time_frame == "M5":
    time_frame_int = '5'
elif time_frame == "M15":
    time_frame_int = '15'
elif time_frame == "M30":
    time_frame_int = '30'
elif time_frame == "H1":
    time_frame_int = '60'
elif time_frame == "H4":
    time_frame_int = '240'
elif time_frame == "D1":
    time_frame_int = '1440'
elif time_frame == "W1":
    time_frame_int = '10080'
else:
    time_frame_int = '43200'

csv_opt = path_opt.replace(time_frame, time_frame_int)

try:
    reports = glob.glob("./{}/Backtest/*.htm".format(path_opt))
    df_sets_info = pd.read_csv("./{}/{}.csv".format(path_opt, csv_opt), sep=';', index_col='Magic_Number')
    df_sets_info.index = df_sets_info.index.astype(str)
except:
    print("Warning: No data for this batch")
    sys.exit()

print("Reading Backtests...")
#import pandas as pd
for i in range(0, len(reports)):  # len(reports)

    magic_number = reports[i].split("Backtest")[-1].split("_")[1]
    file = pd.read_html(reports[i])
    backtest_df = file[1]
    backtest_df.columns = backtest_df.iloc[0]
    backtest_df = backtest_df.iloc[1:]
    backtest_df['Magic_Number'] = magic_number
    if (i == 0):
        bt_df = backtest_df
    else:
        bt_df = bt_df.append(backtest_df)
print("Done")

magics_df = bt_df[['Tiempo', 'Magic_Number', 'Beneficios']].sort_values(['Magic_Number', 'Tiempo'])
magics_df['Year_Month'] = magics_df.Tiempo.dt.strftime('%Y-%m')
magics_df.drop('Tiempo', axis = 1, inplace= True)
magics_df = pd.DataFrame(magics_df.groupby(['Magic_Number', 'Year_Month'])['Beneficios'].sum())
magics_df['Perdida'] = [1 if x < 0 else 0 for x in magics_df['Beneficios']]
magics_df['12_months'] = magics_df['Perdida'].rolling(12).sum()
magics_df['12_months'] = magics_df['12_months'].fillna(0)
magics_df.reset_index(inplace=True)
magic_group_df = magics_df[['Magic_Number', '12_months']].groupby('Magic_Number')
magic_filter_df = magic_group_df.max()
magic_filter_df = magic_filter_df[magic_filter_df['12_months'] < 9]
magic_fil_list = magic_filter_df.index

bt_df = bt_df[bt_df['Magic_Number'].isin(magic_fil_list)].reset_index(drop = True)
bt_df = bt_df[bt_df['Balance'].notna()].reset_index(drop=True)
bt_df['Tiempo'] = pd.to_datetime(bt_df['Tiempo'])
bt_df['Beneficios'] = pd.to_numeric(bt_df.Beneficios)
bt_df['Balance'] = pd.to_numeric(bt_df.Balance)
df_backtest = bt_df.copy()
df_backtest.rename(columns={'Tiempo': 'Close_Date', 'Beneficios': 'Profit',
                            'Volumen': 'Units/Lots'}, inplace=True)
df_backtest['Open_Date'] = df_backtest.Close_Date
df_backtest['Action'] = 'Buy'
df_backtest['Pips'] = 100

# Creating all info needed
# Instancia
df_ins = stat_tools(df_backtest)

# DFs with de info
df_stats = df_ins.get_stat()

# Deleting sets with r2 lower than 0.5
df_stats = df_stats[df_stats.R2_KPI >= 0.95]

# Deleting sets with DD_duration more than 90 days
df_stats = df_stats[df_stats.DD_duration < 365]

df_series = bt_df[['Tiempo', 'Magic_Number', 'Beneficios']]
df_series = df_series[df_series['Magic_Number'].isin(df_stats.index)]

df_series['Tiempo'] = df_series['Tiempo'].dt.strftime('%Y-%m-%d')
df_series = df_series.groupby(['Tiempo', 'Magic_Number'], as_index=False)['Beneficios'].sum().reset_index()

df_pivot = df_series.pivot(index=['index', 'Tiempo'], columns='Magic_Number', values='Beneficios').fillna(
    0).reset_index()
df_pivot = df_pivot.drop(['index'], axis=1)
corr = df_pivot.groupby('Tiempo').sum().corr().abs()
weights_dict = {'Exp_Profit': 1, 'Profit_per_Day': 1, 'PF_nor': 1, 'R2_KPI': 1, 'Win/DD_Ratio_Month': 1,
                'Max_Profit': 1, 'Max_DD': 1, 'DD_Duration': 2, 'Bkn_KPI': 1, 'Sharpe_Ratio': 1, 'SQN': 1,
                'Win_divergence': 0}
demo_sets = []
demo_sets, df_ins = portfolio_creation(df_backtest, demo_sets, weights_dict, corr, max_sets=len(df_backtest),
                                       treshold=0.75)

df_to_demo = df_stats[df_stats.index.isin(demo_sets)][['Max_DD']]
df_lots = df_backtest[df_backtest.Magic_Number.isin(demo_sets)]
df_to_demo['Max_trade_loss'] = df_lots.groupby('Magic_Number')['Profit'].min()
df_to_demo['Mean_trade_loss'] = df_lots[df_lots.Profit < 0].groupby('Magic_Number')['Profit'].mean()
df_to_demo['Lots_dd'] = abs(round((dd_allow * 0.1 / df_to_demo.Max_DD), 2))
df_to_demo['Lots_max_los'] = abs(round((max_loss_allow * 0.1 / df_to_demo.Max_trade_loss), 2))
df_to_demo['Lots_mean_loss'] = abs(round((mean_loss_allow * 0.1 / df_to_demo.Mean_trade_loss), 2))
df_to_demo

pd.set_option('display.max_columns', None)
df_sets_info = df_sets_info[df_sets_info.index.isin(demo_sets)]
df_sets_info[lots_var_name] = df_to_demo[['Lots_dd', 'Lots_max_los', 'Lots_mean_loss']].min(axis=1)

if (not os.path.isdir('./sets_to_demo')):
    os.makedirs('./sets_to_demo')
df_sets_info.to_csv("./sets_to_demo/{}.csv".format(path_opt), sep=";", index=False)