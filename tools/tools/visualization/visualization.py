from random import randint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.stat_tools.stat_tools import stat_tools
import pandas as pd


def sets_color(sets_list):
    color = []
    for i in range(1, (len(sets_list) + 1)):
        color.append('#%06X' % randint(0, 0xFFFFFF))

    return color


class sets_visualization:

    def __init__(self, df, sets_list):

        self.df_ = df
        self.sets_list_ = sets_list
        self.color_list_ = sets_color(sets_list)

    def evolution_plot(self, df_port, df_magic, df_resumen, color_list=None, plot_pf=True):
        """Function to plot sets evolution, month profit and PF evolution, optional
        params:
            df_port = df witth portfolio ecolution
            df_magic = df with magic numbers evolution
            df_resumen = df with profit per month by set
            color_list = list of color to be plot
            set_list = list of sets to plot
            plot_pf = fralg to plot profit factor
        output"""

        sets_list = self.sets_list_
        if color_list is None:
            color_list = self.color_list_

        fig = go.Figure()

        if (plot_pf):
            fig = make_subplots(rows=2, cols=1,
                                specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=df_port.Close_Date,
            y=df_port.Profit_cum,
            name='Portfolio',
            line_color='white',
            visible='legendonly'),
            row=1, col=1)

        for i in range(0, len(sets_list)):

            df_filter = df_magic[df_magic.Magic_Number == sets_list[i]]
            fig.add_trace(go.Scatter(
                x=df_filter.Close_Date,
                y=df_filter.Profit_cum,
                name=str(sets_list[i]),
                line_color=color_list[i],
                opacity=0.8,
                visible='legendonly'), secondary_y=False, )

            if (plot_pf):
                df_filter.loc[df_filter.Profit_Factor_cum > 2, 'Profit_Factor_cum'] = 2.1
                fig.add_trace(go.Scatter(
                    x=df_filter.Close_Date,
                    y=df_filter.Profit_Factor_cum,
                    name=str(sets_list[i]),
                    marker_color=color_list[i],
                    visible='legendonly'),
                    row=2, col=1)

            df_filter = df_resumen[df_resumen.Magic_Number == sets_list[i]]
            fig.add_trace(go.Bar(
                x=df_filter.Year_Month,
                y=df_filter.Norm_profit,
                name=str(sets_list[i]),
                marker_color=color_list[i],
                visible='legendonly'), secondary_y=True, )
        fig.update_layout(
            template="plotly_dark")

        return fig

    def plot_dd(self, color_list=None):
        """Function to plot portfolio and sets DD
        params:
            df = df with sets info
            sets_list = list of sets to plot"""

        df = self.df_
        sets_list = self.sets_list_
        if color_list is None:
            color_list = self.color_list_

        df_ins = stat_tools(df)
        df_port_dd = df_ins.max_profit_dd(df[df.Magic_Number.isin(sets_list)])[2]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_port_dd.Close_Date,
            y=-df_port_dd.DD,
            name='Portfolio',
            line_color='white',
            opacity=0.5,
            fill='tozeroy'))

        for i in range(0, len(sets_list)):
            df_magic_dd = df_ins.max_profit_dd(df[df.Magic_Number == sets_list[i]])[2]
            fig.add_trace(go.Scatter(
                x=df_magic_dd.Close_Date,
                y=-df_magic_dd.DD,
                name=str(sets_list[i]),
                line_color=color_list[i],
                visible='legendonly',
                opacity=0.5))
        fig.update_layout(
            template="plotly_dark")

        return fig

    def plot_regression(self, color_list=None):

        """Function to sets regression
        params:
            df = df with sets info
            sets_list = list of sets to plot"""

        df = self.df_
        sets_list = self.sets_list_
        if color_list is None:
            color_list = self.color_list_

        df_ins = stat_tools(df)
        fig = go.Figure()

        for i in range(0, len(sets_list)):
            df_to_plot = df[df.Magic_Number == sets_list[i]]
            if len(df_to_plot) > 0:
                df_r2 = df_ins.r_squared(df_to_plot)[1]
                fig.add_trace(go.Scatter(
                    x=df_r2.index,
                    y=df_r2.Profit_cum,
                    name=str(sets_list[i]),
                    visible='legendonly',
                    line_color=color_list[i]))
                fig.add_trace(go.Scatter(
                    x=df_r2.index,
                    y=df_r2.Regression,
                    name=str(sets_list[i]) + '_regre',
                    visible='legendonly',
                    line_color='white'))

        fig.update_layout(
            template="plotly_dark")

        return fig

    def month_profit_heatmap(self, df_resumen):
        """Function to plot portfolio and sets DD
        params:
            df_resumen = df with profit per month by set"""

        sets_list = self.sets_list_
        column_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        df_resumen[['Year', 'Month_number']] = df_resumen.Year_Month.str.split("-", expand=True)
        df_resumen['Year'] = df_resumen['Year'].astype('str')
        df_resumen['Month_letter'] = pd.to_datetime(df_resumen['Month_number'], format='%m').dt.strftime('%b')

        fig = go.Figure()

        for i in range(0, len(sets_list)):
            df_to_plot = df_resumen[df_resumen.Magic_Number == sets_list[i]]
            if len(df_to_plot) <= 1:
                continue
            zero_pos = 0 - min(df_to_plot.Profit) / (max(df_to_plot.Profit) - min(df_to_plot.Profit))
            if zero_pos > 0 and  zero_pos < 1:
                color_list = [[0, 'red'], [zero_pos, 'white'], [1, 'green']]
            elif zero_pos < 0:
                color_list = [[0, 'white'], [1, 'green']]
            else:
                color_list = [[0, 'red'], [1, 'white']]

            df_to_plot = df_to_plot.pivot(index='Month_letter', columns='Year', values='Profit')

            df_to_plot = df_to_plot.reindex(column_order, axis=0)

            #df_to_plot.dropna(inplace=True)

            fig.add_trace(
                go.Heatmap(
                    z=df_to_plot.T.values,
                    x=df_to_plot.index,
                    y=df_to_plot.columns,
                    visible=True,
                    hoverinfo='z',
                    colorscale=color_list
                )
            )

        button_list = [
            {'label': col, 'method': 'update',
             'args': [{'visible': [True if x == col else False for x in sets_list]}]}
             for col in sets_list]

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(type='category'),
            updatemenus=[
                dict(
                    active=0,
                    buttons=list(button_list),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1,
                    xanchor="right",
                    y=1.05,
                    yanchor="bottom"
                )
            ])

        return fig

