import datetime

from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.read_csv('CENTRALIZED_data_record.csv')

df['Time'] = pd.to_datetime(df['Time'])
df['Power'] = pd.DataFrame.abs(df['Voltage'] * df['Current'])

fig = px.line(df, x="Time", y=["Voltage", "Current", "Power"])

app.layout = html.Div(children=[
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),
    html.H4(children='Estimated Bill Cost'),
    html.H1(children='10000000'),
])

@app.callback(Output('example-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_line_graph(n):
    df2 = pd.read_csv('CENTRALIZED_data_record.csv')
    df2['Time'] = pd.to_datetime(df['Time'])
    df2['Power'] = pd.DataFrame.abs(df['Voltage'] * df['Current'])
    fig = px.line(df, x="Time", y=["Voltage", "Current", "Power"])
    return(fig)


if __name__ == '__main__':
    app.run_server(debug=True,port=8080,host='192.168.1.4')
