import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import dash_uploader as du
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
du.configure_upload(app, UPLOAD_FOLDER)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Interactive ML Dashboard"),
    
    html.H3("Upload Your Dataset (CSV Format)"),
    du.Upload(id='upload-data', text='Drag and Drop or Select Files',
              text_completed='Uploaded: {filename}', cancel_button=True),
    
    html.Br(),
    html.Label("Select Target Column for Prediction:"),
    dcc.Dropdown(id='target-column', placeholder="Choose target column"),
    
    html.Br(),
    html.Button("Train Model", id='train-button', n_clicks=0),
    html.H3(id='model-output', children=""),
    
    html.H3("Data Visualization"),
    dcc.Graph(id='scatter-plot')
])

# Store the uploaded dataset
dataset_store = {}

# Callback for file upload
@app.callback(
    Output('target-column', 'options'),
    Input('upload-data', 'isCompleted'),
    Input('upload-data', 'fileNames')
)
def update_dropdown(isCompleted, fileNames):
    if isCompleted and fileNames:
        file_path = os.path.join(UPLOAD_FOLDER, fileNames[0])
        dataset = pd.read_csv(file_path)
        dataset_store['data'] = dataset  # Store dataset globally
        return [{'label': col, 'value': col} for col in dataset.columns]
    return []

# Callback to train model
@app.callback(
    Output('model-output', 'children'),
    Input('train-button', 'n_clicks'),
    Input('target-column', 'value')
)
def train_model(n_clicks, target_column):
    if n_clicks > 0 and 'data' in dataset_store and target_column:
        dataset = dataset_store['data']
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        # Handle categorical data
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return f'Model trained with R² score: {score:.4f}'
    return ""

# Callback for visualization
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('target-column', 'value')
)
def update_graph(target_column):
    if 'data' in dataset_store and target_column:
        dataset = dataset_store['data']
        fig = px.scatter_matrix(dataset, dimensions=dataset.columns, color=target_column)
        return fig
    return px.scatter(title="Upload a dataset to visualize")

if __name__ == '__main__':
    app.run(debug=True)
