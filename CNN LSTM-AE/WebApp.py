import datetime

from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div(children=[
    html.Label('Duration'),
    dcc.Dropdown(
        id='duration',
        options=[
            {'label': '1 hour', 'value':'1 hour'},
            {'label': '2 hours', 'value':'2 hours'},
            {'label': '4 hours', 'value':'4 hours'},
            {'label': '6 hours', 'value':'6 hours'},
        ],
        value='1 hour'
    ),
    
    dcc.Graph(
        id='example-graph',
    ),

    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),
])

@app.callback(Output('example-graph', 'figure'),
              Input('interval-component', 'n_intervals'),
             Input('duration', 'value'))
def update_line_graph(n, selected_duration):
    df = pd.read_csv('web_record.csv', names=["Time", "Voltage", "Current", "Type", "Prediction"])
    duration = 360 - int(selected_duration[:-5]) * 60
#     features, labels = prepareData('data_record.csv', 7)
#     duration = 60
#     scaler = MinMaxScaler()
#     print(features)
#     for i in range(duration):
#         normFeatures = scaler.fit_transform(features)
#         interpreter = tf.lite.Interpreter(model_path="liteKitchen.tflite")
#         interpreter.allocate_tensors()
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
#         x_tensor = np.expand_dims(normFeatures, axis=0).astype(np.float32)
#         interpreter.set_tensor(input_details[0]['index'], x_tensor)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         output_data = (output_data[0][0] * (maxPower - minPower)) + minPower
        
        
#     for i in range(len(features)):
#         aveVoltage += features[i][0]
#         aveCurrent += features[i][1]
#     aveVoltage = aveVoltage / len(features)
#     aveCurrent = aveCurrent / len(features)
#     
#     for i in range(duration):
#         scaler = MinMaxScaler()
#         normFeatures = scaler.fit_transform(features)
        
    
    df['Time'] = pd.to_datetime(df['Time'])
    df['Power'] = pd.DataFrame.abs(df['Voltage'] * df['Current'])
    fig = px.line(df, x="Time", y=["Voltage", "Current", "Power", "Prediction"])
    fig['layout']['uirevision'] = 42
    return (fig)


if __name__ == '__main__':
    app.run_server(debug=False,port=8080,host='192.168.1.157')
