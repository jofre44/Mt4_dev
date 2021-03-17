import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import sklearn.metrics as sk
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None



def magic_correlation(df, magic, treshold=0.5):
    """Select sets that have less correlation than treshold
    params:
        df = correlation df
        magic = set of reference
        treshold = value to filter
    output = list of sets"""

    df_corr = df[[magic]][df[magic] <= treshold].sort_values(by=magic)
    magic_list = list(df_corr.index)
    return magic_list


def possible_portfolio(corr_dict, magic_list):
    """Create all the possible portfolio for a prev list of sets
    param:
        corr_dict = a dictionary where the keys are the pre sets selectes and the values is a list of posible sets to add
    output = dict where keys are the portfolio and values is a list of the sets in portfolio"""

    repeat_magic_list = [*corr_dict.values()]
    repeat_magic_list = np.concatenate(repeat_magic_list)
    repeat_magic_list = pd.Series(repeat_magic_list).value_counts()
    repeat_magic_list = repeat_magic_list[repeat_magic_list > (len(magic_list) - 1)].index.tolist()
    magic_for_port_dict = {'magic_for_port': repeat_magic_list}
    port_dict = {id_port: list(corr_dict.keys()) + [magic] for id_port, magic in
                 enumerate(magic_for_port_dict['magic_for_port'], 1)}

    return port_dict

def portfolio_creation(df, magic_list, weights_dict, corr, max_sets=30, treshold=0.5):
    """Function to create portfolio
    param:
        df = dataframe with set to analyze
        magic_list = list of magic number in the portfolio. Could be empty
        weights_dict = dict with kpi weights for select set to add a portfolio
        corr = correlation between sets
        max_sets = maximum sets in portfolio
        treshold = maximum correlatio posible to add a set in portfolio
    output = list with magic numbers to add"""

    scaler = MinMaxScaler()
    # Instance
    df_ins = stat_tools(df)

    df_weight = pd.Series(weights_dict)

    while (len(magic_list) < max_sets):
        if len(magic_list) == 0:
            print("\n---------- New Portfolio")
            df_stats = df_ins.get_stat(portfolio_creation=True)
            df_scale = pd.DataFrame(scaler.fit_transform(df_stats), columns=df_stats.columns).set_index(df_stats.index)
            df_scale['DD_Duration'] = 1 - df_scale['DD_Duration']
            df_scale['kpis_sum'] = (df_scale * df_weight).sum(1)
            port_id = df_scale['kpis_sum'].idxmax()
            print("\n-- Set {} with max kpi value: {}".format(port_id, max(df_scale['kpis_sum'])))
            print("\n-- Portfolio DD : {}".format(df_stats.loc[df_stats.index == port_id, 'Max_DD'].values[0]))
            magic_list = [df_scale['kpis_sum'].idxmax()]
        else:
            print("\n---------- Adding new set, actual portfolio with {} sets".format(len(magic_list)))
            corr_dict = {magic: magic_correlation(corr, magic, treshold) for magic in magic_list}
            port_dict = possible_portfolio(corr_dict, magic_list)
            if not port_dict:
                print("\n-- Not more sets to add ")
                break
            df_port = pd.DataFrame()
            for port_id in port_dict.keys():
                df_test = df_ins.get_stat(port_id, df[df.Magic_Number.isin(port_dict[port_id])],
                                          portfolio_creation=True)
                df_port = df_port.append(df_test)
            df_scale = pd.DataFrame(scaler.fit_transform(df_port), columns=df_port.columns).set_index(df_port.index)
            df_scale['DD_Duration'] = 1 - df_scale['DD_Duration']
            df_scale['kpis_sum'] = (df_scale * df_weight).sum(1)
            port_id = df_scale['kpis_sum'].idxmax()
            print("\n-- Portfolio {} with max kpi value: {}".format(port_id, max(df_scale['kpis_sum'])))
            print("\n-- Portfolio DD : {}".format(df_port.loc[df_port.index == port_id, 'Max_DD'].values[0]))
            magic_list = port_dict[df_scale['kpis_sum'].idxmax()]

    return magic_list, df_ins

class stat_tools:
    """Class to make the resume of sets in the account"""

    def __init__(self, df, ini_balance = 10000):
        self.obj_ = df
        self.balance = ini_balance

    def profit_trades(self, df = None):
        """Fuunction to calculate trades and profit for each set in account"""

        if df is None:
            df = self.obj_
        profit = df.Profit.sum().round(2)
        trades = df.Profit.count()
        exp_profit = round(profit/ trades,2)
        
        min_date = min(df.Open_Date)
        max_date = pd.to_datetime('today', format='%Y-%m-%d')
        months =  (max_date.year - min_date.year) * 12 + max_date.month - min_date.month
        trades_month = trades/months

        return (profit, trades_month, exp_profit, trades, months)

    def days_in_real(self, df = None):
        """How long have been the set operating"""

        if df is None:
            df = self.obj_
        df['Open_Date'] = pd.to_datetime(df['Open_Date'])
        today_date = pd.to_datetime('today', format='%Y-%m-%d')
        days_operating = (today_date - min(df['Open_Date'])).days

        return days_operating

    def normalize_profit(self, df = None):
        """Normalize profit diving by lots in trade"""

        if df is None:
            df = self.obj_
        df['Units/Lots'] = df['Units/Lots'].astype(float)
        df['Profit_norm'] = df['Profit'] / (df['Units/Lots'] * 100)

        return df

    def profit_factor(self, df = None, normalize=False):
        """Calculate profit factor. It can be calculate with original data or
        normalizing the profit with respect the lots"""

        if df is None:
            df = self.obj_
        if (normalize):
            df = self.normalize_profit(df)
            profit_pos = df[df.Profit > 0]['Profit_norm'].sum()
            profit_neg = df[df.Profit < 0]['Profit_norm'].sum()
        else:
            profit_pos = df[df.Profit > 0]['Profit'].sum()
            profit_neg = df[df.Profit < 0]['Profit'].sum()
        profit_factor = round(abs(profit_pos / profit_neg), 2)

        return (profit_factor)

    def max_profit_dd(self, df  = None):
        """Calculate max profit and DD for set"""

        if df is None:
            df = self.obj_
        df['Open_Date'] = pd.to_datetime(df['Open_Date'])
        df['Close_Date'] = pd.to_datetime(df['Close_Date'])
        df = df.sort_values(by='Close_Date')
        df['Profit_cum'] = df.Profit.cumsum(skipna=True)
        df['Actual_max'] = df.Profit_cum.cummax(skipna=True)
        df.loc[df.Actual_max < 0, 'Actual_max'] = 0
        df['DD'] = df.Profit_cum - df.Actual_max
        max_profit = round(max(df.Actual_max), 2)
        max_dd = round(min(df.DD), 2)
        bkn_kpi = round(sum(df.DD), 2)
        dd_percentage = abs(max_dd*100/self.balance)

        return (max_profit, max_dd, df, bkn_kpi, dd_percentage)

    def longest_dd_duration(self, df = None):
        """To  calculate the longest dd"""

        if df is None:
            df = self.obj_
        df= self.max_profit_dd(df)[2]

        df = df[['Close_Date', 'Profit_cum', 'Actual_max']]
        date_time_last = df.iloc[-1]['Close_Date']
        df = df.append(pd.DataFrame({'Close_Date': pd.to_datetime('today', format='%Y-%m-%d'),
                                     'Profit_cum': df.loc[df.Close_Date == date_time_last, 'Profit_cum'],
                                     'Actual_max': df.loc[df.Close_Date == date_time_last, 'Actual_max']}),
                       ignore_index=True)
        df = df.groupby(['Actual_max'], as_index=False)['Close_Date'].agg(['min', 'max']). \
            rename(columns={'min': 'min_date',
                            'max': 'max_date'})
        df = df[(df[['min_date', 'max_date']] != 0).all(axis=1)]
        df['Diff'] = (df.max_date - df.min_date).dt.days

        return max(df.Diff)

    def r_squared(self, df = None):
        """Calculate r2 """

        if df is None:
            df = self.obj_
        df['Close_Date'] = pd.to_datetime(df['Close_Date'])
        df = df.sort_values(by=['Close_Date'])
        df['Profit_cum'] = df.Profit.cumsum(skipna = True)
        df = df[['Close_Date', 'Profit_cum']].set_index('Close_Date')
        df['Numeric_date'] = mdates.date2num(df.index)
        reg = np.polyfit(np.array(df.Numeric_date), np.array(df.Profit_cum),1)
        df['Regression'] = reg[0] * df.Numeric_date + reg[1]
        r2_kpi = round(sk.r2_score(df.Profit_cum, df.Regression),2)

        return (r2_kpi, df)

    def month_win_dd_ratio(self, df = None):
        """Calculcate the win/ DD ratio per month"""

        if df is None:
            df = self.obj_
        
        anual_return = 0.15
        max_dd_allow = 0.12
        month_numbers = len(self.profit_per_period(df))
        
        min_win_dd_ratio = (anual_return/12*month_numbers)/max_dd_allow      
        max_dd =  abs(self.max_profit_dd(df)[1])
        profit_current = self.profit_trades(df)[0]
        rate_win_dd = round(profit_current/(max_dd), 4)
        rate_win_dd_percent = rate_win_dd/min_win_dd_ratio

        return rate_win_dd_percent

    def sharpe_ratio_sqn(self, df=None):
        """Calculate Sharpe Ratio and  SQN"""
        if df is None:
            df = self.obj_

        profit_mean = df.Profit.mean()
        profit_std = df.Profit.std()
        oper_num = len(df)
        sharpe_ratio = (profit_mean/profit_std)*(252**0.5)
        sqn = (profit_mean/profit_std)*(oper_num**0.5)

        return(sharpe_ratio, sqn)

    def win_divergence(self, df=None):

        """Calculate win divergence"""
        if df is None:
            df = self.obj_

        df = self.normalize_profit(df)

        df['Profit_neg'] = np.where(df['Profit_norm'] < 0, 1, 0)
        df['Profit_pos'] = np.where(df['Profit_norm'] > 0, 1, 0)

        total_neg_trades = df.Profit_neg.sum()
        total_pos_trades = df.Profit_pos.sum()
        win_trades_ratio = round(total_pos_trades / (total_neg_trades + total_pos_trades), 2)

        df['Cons_neg_trades'] = df.Profit_neg * (df.Profit_neg.groupby(
            (df.Profit_neg != df.Profit_neg.shift()).cumsum()).cumcount() + 1)
        df['Cons_neg_shift'] = df.Cons_neg_trades.shift(-1).fillna(0)
        df['Cons_neg_max'] = np.where(df.Cons_neg_trades > df.Cons_neg_shift,
                                      df.Cons_neg_trades, 0)
        df['Cons_pos_trades'] = df.Profit_pos * (df.Profit_pos.groupby(
            (df.Profit_pos != df.Profit_pos.shift()).cumsum()).cumcount() + 1)
        df['Cons_pos_shift'] = df.Cons_pos_trades.shift(-1).fillna(0)
        df['Cons_pos_max'] = np.where(df.Cons_pos_trades > df.Cons_pos_shift,
                                      df.Cons_pos_trades, 0)
        max_cons_loss = df.Cons_neg_max.max()
        max_cons_win = df.Cons_pos_max.max()
        mean_cons_loss = df[df.Cons_neg_max > 0]['Cons_neg_max'].mean()
        mean_cons_win = df[df.Cons_pos_max > 0]['Cons_pos_max'].mean()
        mean_loss = abs(round(df[df.Profit_neg == 1]['Profit_norm'].mean(), 2))
        mean_win = round(df[df.Profit_pos == 1]['Profit_norm'].mean(), 2)
        total_pos_trades = df.Profit_pos.sum()
        total_neg_trades = df.Profit_neg.sum()

        success_rate = round(total_pos_trades/(total_neg_trades+total_pos_trades),2)
        loss_win_rate = round(mean_loss / (mean_loss + mean_win), 2)
        dw_kpi = round(win_trades_ratio - loss_win_rate, 2) * 100

        return dw_kpi, max_cons_loss, mean_cons_loss, max_cons_win, mean_cons_win, success_rate, loss_win_rate, win_trades_ratio

    def profit_per_period(self, df = None):
        """Calculate the profit por period. Harcode to calculate profite per year-month
        TODO: generalize to calculate profit for a given period"""
        if df is None:
            df = self.obj_
            group_list = ['Magic_Number', 'Year_Month']
        else:
            group_list = [ 'Year_Month']
        df['Close_Date'] = pd.to_datetime(df['Close_Date'])
        df['Year_Month'] = df['Close_Date'].dt.strftime('%Y-%m')
        df_month_profit = df.groupby(['Magic_Number', 'Year_Month'], as_index=False)['Profit'].sum()
        df_month_profit['Abs_profit'] = df_month_profit.Profit.abs()
        df_month_profit['Max_value'] = df_month_profit.groupby('Magic_Number')['Abs_profit'].transform('max')
        df_month_profit['Norm_profit'] = df_month_profit.Profit / df_month_profit.Max_value

        return df_month_profit

    def set_magic_evolution(self):
        """Calculate the evolution of the profit and PF for each magic"""

        df = self.obj_
        df = df[['Magic_Number', 'Close_Date', 'Units/Lots', 'Action', 'Pips', 'Profit']].sort_values(
            by=['Magic_Number', 'Close_Date'])
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        df = self.normalize_profit(df)
        df['Profit_pos'] = np.where(df['Profit_norm'] > 0, df.Profit, 0)
        df['Profit_neg'] = np.where(df['Profit_norm'] < 0, df.Profit, 0)
        df = df.fillna(0)
        df['Profit_cum'] = df.groupby(['Magic_Number'])['Profit'].apply(lambda x: x.cumsum(skipna=True))
        df['Profit_cum_neg'] = df.groupby(['Magic_Number'])['Profit_neg'].apply(lambda x: x.cumsum(skipna=True))
        df['Profit_cum_pos'] = df.groupby(['Magic_Number'])['Profit_pos'].apply(lambda x: x.cumsum(skipna=True))
        df['Profit_Factor_cum'] = df['Profit_cum_pos'] / df['Profit_cum_neg'].abs()
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        df = df.round(2)

        return df

    def set_port_evolution(self, df = None):
        """Calculate the evolution of the profit and PF for port
        param:
        df = to create evolution of a sub set"""

        if df is None:
            df = self.obj_
        df = df[['Magic_Number', 'Close_Date', 'Units/Lots', 'Action', 'Pips', 'Profit']].sort_values(
            by=['Close_Date'])
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        df = self.normalize_profit(df)
        df['Profit_pos'] = np.where(df['Profit_norm'] > 0, df.Profit, 0)
        df['Profit_neg'] = np.where(df['Profit_norm'] < 0, df.Profit, 0)
        df = df.fillna(0)
        df['Profit_cum'] = df.Profit.cumsum(skipna=True)
        df['Profit_cum_neg'] = df.Profit_neg.cumsum(skipna=True)
        df['Profit_cum_pos'] = df.Profit_pos.cumsum(skipna=True)
        df['Profit_Factor_cum'] = df['Profit_cum_pos'] / df['Profit_cum_neg'].abs()
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        df = df.round(2)

        return df
        
    def get_stat(self, portfolio = None, df = None, portfolio_creation = False):
        """Create dataframe with the basic stat of a portfolio
        params:
        portfolio = Name of portolio
        df = None for stat of all data in the instance, else give the sub data"""

        if df is None:
            df = self.obj_

        if portfolio is None:
            df_grp = df.groupby(['Magic_Number'])
            dic_stat = {magic: self.get_stat_dict(portfolio_creation, df_grp.get_group(magic))
                        for magic in df_grp.groups.keys()}
        else:
            dic_stat = {portfolio: self.get_stat_dict(portfolio_creation, df)}

        if not portfolio_creation:
            df_stat = pd.DataFrame.from_dict(dic_stat, orient='index').sort_values(by='Profit', ascending=False)
        else:
            df_stat = pd.DataFrame.from_dict(dic_stat, orient='index')

        if portfolio is None:
            df_stat.index.name = 'Magic_number'
        else:
            df_stat.index.name = 'portfolio'

        return df_stat

    def get_stat_dict(self, portfolio_creation, df = None):
        """Create dic with the basic stats"""
        if df is None:
            df = self.obj_

        if portfolio_creation:

            profit, trades_months, exp_profit, trades, months = self.profit_trades(df)
            r2_kpi, _ = self.r_squared(df)
            sharpe_ratio, sqn = self.sharpe_ratio_sqn(df)
            max_profit, max_dd, _, bkn_kpi, _= self.max_profit_dd(df)
            dw_kpi, max_cons_loss, mean_cons_loss, _, _, _, _, _ = self.win_divergence(df)

            dic_stat = {'Exp_profit': exp_profit,
                        'Profit_per_Day': round(profit/self.days_in_real(df),2),
                        'PF_Nor': self.profit_factor(df, normalize=True),
                        'R2_KPI': r2_kpi,
                        'Win/DD_Ratio_Month': self.month_win_dd_ratio(df),
                        'Win_diverngence': dw_kpi,
                        'Sharpe_Ratio': sharpe_ratio,
                        'SQN': sqn,
                        'Max_Profit': max_profit,
                        'Max_DD': max_dd,
                        'DD_Duration': self.longest_dd_duration(df),
                        'Bkn_KPI': bkn_kpi}
        else:

            profit, trades_month, exp_profit, trades, months = self.profit_trades(df)
            r2_kpi, _ = self.r_squared(df)
            sharpe_ratio, sqn = self.sharpe_ratio_sqn(df)
            max_profit, max_dd, _, bkn_kpi, dd_percentage = self.max_profit_dd(df)
            dw_kpi, max_cons_loss, mean_cons_loss, _, _, _, _, _ = self.win_divergence(df)

            dic_stat = {'Profit': profit,
                        'Months': months,
                        'Trades/Month': trades_month,
                        'Exp_Profit' : exp_profit,
                        'PF': self.profit_factor(df),
                        'PF_Nor': self.profit_factor(df, normalize=True),
                        'Max_DD_%': dd_percentage,
                        'R2_KPI': r2_kpi,
                        'Win/DD_Ratio': self.month_win_dd_ratio(df),
                        'Mean_cons_loss': mean_cons_loss,
                        'Max_cons_loss': max_cons_loss,
                        'Win_divergence': dw_kpi,
                        'Max_Profit': max_profit,
                        'Max_DD': max_dd,
                        'DD_duration': self.longest_dd_duration(df)}

        return dic_stat

    def get_lots(self, df=None):

        """Create dataframe with the relotation of each set to add in a portfolio
                params:
                df = None for stat of all data in the instance, else give the sub data"""

        if df is None:
            df = self.obj_

        df_grp = df.groupby(['Magic_Number'])
        dic_stat = {magic: self.get_lots_dict(df_grp.get_group(magic))
                    for magic in df_grp.groups.keys()}

        df_stat = pd.DataFrame.from_dict(dic_stat, orient='index')
        df_stat.index.name = 'Magic_number'

        return df_stat

    def get_lots_dict(self, df=None):
        """Create dic with the basic stats"""
        if df is None:
            df = self.obj_

        df['Units/Lots'] = df['Units/Lots'].astype(float)
        act_lots = df['Units/Lots'].max()
        win_trades_ratio = self.win_divergence(df)[7]
        reloted_lot = round(win_trades_ratio*act_lots,2)

        dic_stat = {
            'Backtest_Lots': act_lots,
            'Win_Percent': win_trades_ratio,
            'Lots_per_WP':reloted_lot
        }

        return dic_stat
