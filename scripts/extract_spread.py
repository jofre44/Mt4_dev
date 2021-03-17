import darwinex_ticks as dtw1
import pandas as pd
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


spreads_info = pd.DataFrame(columns=['Pair', 'Spread'])

total_pair = pd.DataFrame(dwt._ftpObj.nlst()[2:])[0]

for pair in total_pair:
    spread_factor=spreads_factors[spreads_factors.Pair==pair].Spread_F
    spread_info = pd.DataFrame()
    
    try:
        data = dwt.ticks_from_darwinex(pair, start=start_date, end=end_date)
    except:
        print(["Error in symbol "+ pair])
        continue
    
    data_spread= dtw1.spread(data)
    
    try:
        spread_pair = [round((data_spread*spread_factor.iloc[0]).quantile(percentile),5)]
    except:
        spread_pair = ['not in spread factor´s file, data spread : '+ str(round(np.max(data_spread),5))]
        
    print(pair," = ", spread_pair)
    spread_info = pd.DataFrame({'Pair' : [pair],
                           'Spread' : spread_pair})
    spreads_info = spreads_info.append(spread_info, ignore_index=True)
    

str_pair = ''
for i in range(0, (len(spreads_info))): 
        str_pair = str_pair + spreads_info['Pair'][i] + '='+ str(spreads_info['Spread'][i]) + '\n'

file = open('pairs.txt','w') 
 
file.write(str_pair) 
 
file.close() 