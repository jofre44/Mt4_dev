from tools.stat_tools.stat_tools import *
import pandas as pd
import os
import re
import glob

BASE_DIR = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-4])
DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKTEST_DIR = os.path.join(BASE_DIR, 'backtest')

def reports_list(magic_number):
    path = os.path.join(BACKTEST_DIR, '*.htm')
    reports = glob.glob(path)
    reports = [e for e in reports if re.split(' |_',e)[4] in magic_number]
    return reports


def get_df_stats(magic_number):
    magic_number_aux = magic_number
    magic_number = list(dict.fromkeys(magic_number))
    bt_path = os.path.join(DATA_DIR, 'backtest_stats.csv')
    try:
        df_csv = pd.read_csv(bt_path, sep=",", index_col='Magic_number')
    except:
        df_csv = None

    if df_csv is not None:
        mn_bt = list(map(str, df_csv.index))
        magic_to_add = [e for e in magic_number if e not in mn_bt]
        magic_number = magic_to_add

    df_stats = None
    if len(magic_number) > 0:
        reports = reports_list(magic_number)
        df_bt = read_backtest(magic_number, reports)
        if df_bt is not None:
            df_ins = stat_tools(df_bt)
            df_stats = df_ins.get_stat()

    if (df_csv is not None) and (df_stats is not None):
        df_tot = df_csv.append(df_stats).sort_values(by='Profit', ascending=False)
        df_tot.to_csv(bt_path)
    elif df_stats is not None:
        df_tot = df_stats.sort_values(by='Profit', ascending=False)
        df_tot.to_csv(bt_path)
    else:
        df_tot = df_csv

    df_tot = df_tot[df_tot.index.isin(magic_number_aux)]

    return df_tot


def read_backtest(magic_number, reports=None, verbose = False):
    magic_number = list(dict.fromkeys(magic_number))

    if reports is None:
        reports = reports_list(magic_number)

    bt_df = None
    for i in range(0, len(reports)):
        magic_number = re.split(' |_', reports[i])[4]
        if verbose:
            print('Reading BT: ', magic_number)
        file = pd.read_html(reports[i])
        backtest_df = file[1]
        backtest_df.columns = backtest_df.iloc[0]
        backtest_df = backtest_df.iloc[1:]
        backtest_df['Magic_Number'] = magic_number
        if (i == 0):
            bt_df = backtest_df
        else:
            bt_df = bt_df.append(backtest_df)

    if bt_df is None:
        return None

    bt_df = bt_df[bt_df['Balance'].notna()].reset_index(drop=True)
    bt_df['Tiempo'] = pd.to_datetime(bt_df['Tiempo'])
    bt_df['Beneficios'] = pd.to_numeric(bt_df.Beneficios)
    bt_df['Balance'] = pd.to_numeric(bt_df.Balance)
    bt_df.rename(columns={'Tiempo': 'Close_Date', 'Beneficios': 'Profit',
                          'Volumen': 'Units/Lots'}, inplace=True)
    bt_df['Open_Date'] = bt_df.Close_Date
    bt_df['Action'] = 'Buy'
    bt_df['Pips'] = 100

    return bt_df