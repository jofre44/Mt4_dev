import darwinex_ticks as dtw1
import pandas as pd
import glob, os
import numpy as np

spreads_factors=pd.read_csv("Spreads_factors.txt", header = None, sep = '=', names = ['Pair','Spread_F'])
configs_info=pd.read_csv("configs.txt", header = None, sep= "=", names=['Desc', 'Value'])

user= configs_info[configs_info.Desc=='user'].Value.iloc[0]
password= configs_info[configs_info.Desc=='password'].Value.iloc[0]
host_name= configs_info[configs_info.Desc=='host_name'].Value.iloc[0]
start_date= configs_info[configs_info.Desc=='start_date'].Value.iloc[0]
end_date= configs_info[configs_info.Desc=='end_date'].Value.iloc[0]
percentile= pd.to_numeric(configs_info[configs_info.Desc=='percentile'].Value.iloc[0])

dwt = dtw1.DarwinexTicksConnection(dwx_ftp_user=user,
                       dwx_ftp_pass=password,
                       dwx_ftp_hostname=host_name,
                       dwx_ftp_port=21)

path = r'data\*.csv'
csv_files = reports = glob.glob(path)

print("Total files: {}".format(len(csv_files)))
i = 1
for file in csv_files:
    pair = file.split("\\")[-1][0:6]
    print("\n--- CREATING DATA FOR FILE {}: {} ---\n".format(i, pair))

    # Reading data from darwinex
    print("Dowloading from {} to {}".format(start_date, end_date))
    data = dwt.ticks_from_darwinex('EURUSD', start=start_date, end=end_date)
    data['Price'] = (data.Ask + data.Bid) / 2
    data_ohlc = data['Price'].resample('1Min').ohlc()

    # Date for filtering
    start_date = pd.to_datetime('2006-07-01').tz_localize('UTC')
    min_date = min(data_ohlc.index)
    # min_hour = data_ohlc.Hour[0]
    max_date = max(data_ohlc.index)

    # Reading data from mt4
    print("Reading csv")
    data_mt = pd.read_csv(file, header=None)

    # Procesing data
    data_mt.columns = ["Date", "Hora", "open", "high", "low", "close", "Vol"]
    data_mt['Time'] = pd.to_datetime(data_mt['Date'] + ' ' + data_mt['Hora'])
    data_mt = data_mt[['Time', 'open', 'high', 'low', 'close']].set_index('Time')

    # Setting time zone
    data_mt.index = data_mt.index.tz_localize('UTC')
    data_mt.index = data_mt.index - pd.Timedelta(hours=2)
    data_mt = data_mt.loc[start_date:min_date]

    # Merging data
    data_total = data_mt.append(data_ohlc).reset_index()
    data_total['Date'] = data_total.Time.dt.strftime('%Y.%m.%d')
    data_total['Hour'] = data_total.Time.dt.strftime('%H:%M')
    data_total = data_total[['Date', 'Hour', 'open', 'high', 'low', 'close']]

    # Saving csv data
    print("Saving update data...")
    data_total.to_csv("{}_from_{}_to_{}.csv".format(pair, str(start_date).split(sep=" ")[0],
                                                    str(max_date).split(sep=" ")[0]), header=False,
                      index=False)

    i += 1
    del data, data_ohlc, data_total, data_mt
