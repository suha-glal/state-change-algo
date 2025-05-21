"""
State Change Detection App using Dash (by Plotly)
This application analyzes Empatica wristband data to detect psychological state changes.
"""

import base64
import datetime
import io
import json
import tempfile
import os

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout
app.layout = html.Div([
    dbc.Container([
        html.H1("State Change Detection for Empatica Wristband", className="text-center my-4"),
        html.P("Upload your Empatica wristband data files to detect psychological state changes", 
               className="lead text-center mb-5"),
        
        dbc.Card([
            dbc.CardHeader(html.H4("Upload Data Files")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("EDA Data (CSV)"),
                        dcc.Upload(
                            id='upload-eda',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='eda-upload-status')
                    ], md=6),
                    dbc.Col([
                        html.Label("Pulse Rate Data (CSV)"),
                        dcc.Upload(
                            id='upload-pulse',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='pulse-upload-status')
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Temperature Data (CSV)"),
                        dcc.Upload(
                            id='upload-temp',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='temp-upload-status')
                    ], md=6),
                    dbc.Col([
                        html.Label("Accelerometers Data (CSV)"),
                        dcc.Upload(
                            id='upload-accel',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='accel-upload-status')
                    ], md=6)
                ]),
                
                html.Hr(),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Window Sizes (minutes)"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    id='window-sizes',
                                    options=[
                                        {'label': '3 min', 'value': 3},
                                        {'label': '4 min', 'value': 4},
                                        {'label': '5 min', 'value': 5},
                                        {'label': '6 min', 'value': 6},
                                        {'label': '8 min', 'value': 8},
                                        {'label': '10 min', 'value': 10},
                                        {'label': '15 min', 'value': 15},
                                        {'label': '20 min', 'value': 20},
                                        {'label': '30 min', 'value': 30}
                                    ],
                                    value=[3, 4, 5, 6, 8, 10, 15, 20, 30],
                                    inline=True
                                )
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Select All", id="select-all-btn", color="outline-secondary", size="sm", className="me-2"),
                                dbc.Button("Deselect All", id="deselect-all-btn", color="outline-secondary", size="sm")
                            ], width=12, className="mt-2")
                        ])
                    ], md=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Significance Threshold (standard deviations): ", className="mt-3"),
                        html.Span(id="threshold-value", children="2.0"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=1.0,
                            max=5.0,
                            step=0.1,
                            value=2.0,
                            marks={i: str(i) for i in range(1, 6)}
                        )
                    ], md=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Process Data", id="process-btn", color="primary", className="w-100 mt-4", disabled=True)
                    ], md=12)
                ])
            ])
        ], className="mb-4"),
        
        # Loading spinner
        dbc.Spinner(
            html.Div(id="loading-output"),
            color="primary",
            type="border",
            fullscreen=True,
            fullscreen_style={"backgroundColor": "rgba(0, 0, 0, 0.3)"}
        ),
        
        # Results section
        html.Div(id="results-container", style={"display": "none"}, children=[
            dbc.Card([
                dbc.CardHeader(html.H4("Detected State Changes")),
                dbc.CardBody([
                    html.Div(id="state-changes-table")
                ])
            ], className="mb-4"),
            
            html.H4("Biomarker Data Visualization", className="mb-3"),
            
            dbc.Tabs([
                dbc.Tab(label="EDA", tab_id="tab-eda", children=[
                    dcc.Graph(id="eda-graph", style={"height": "500px"})
                ]),
                dbc.Tab(label="Pulse Rate", tab_id="tab-pulse", children=[
                    dcc.Graph(id="pulse-graph", style={"height": "500px"})
                ]),
                dbc.Tab(label="Temperature", tab_id="tab-temp", children=[
                    dcc.Graph(id="temp-graph", style={"height": "500px"})
                ]),
                dbc.Tab(label="Accelerometers", tab_id="tab-accel", children=[
                    dcc.Graph(id="accel-graph", style={"height": "500px"})
                ])
            ], id="tabs", active_tab="tab-eda")
        ])
    ], fluid=True)
])

# Store uploaded data
app.temp_files = {
    'eda': None,
    'pulse': None,
    'temp': None,
    'accel': None
}

# Store processed data
app.processed_data = None

# Helper function to parse uploaded files
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                temp.write(decoded)
                temp_path = temp.name
            
            return temp_path, True, f"Uploaded: {filename}"
        else:
            return None, False, "Please upload a CSV file."
    except Exception as e:
        return None, False, f"Error processing file: {e}"

# Callback for file uploads
@app.callback(
    [Output('eda-upload-status', 'children'),
     Output('pulse-upload-status', 'children'),
     Output('temp-upload-status', 'children'),
     Output('accel-upload-status', 'children'),
     Output('process-btn', 'disabled')],
    [Input('upload-eda', 'contents'),
     Input('upload-pulse', 'contents'),
     Input('upload-temp', 'contents'),
     Input('upload-accel', 'contents')],
    [State('upload-eda', 'filename'),
     State('upload-pulse', 'filename'),
     State('upload-temp', 'filename'),
     State('upload-accel', 'filename')]
)
def update_output(eda_contents, pulse_contents, temp_contents, accel_contents,
                 eda_filename, pulse_filename, temp_filename, accel_filename):
    
    # Initialize outputs
    eda_status = no_update
    pulse_status = no_update
    temp_status = no_update
    accel_status = no_update
    
    # Check which input triggered the callback
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'upload-eda' and eda_contents:
        temp_path, success, message = parse_contents(eda_contents, eda_filename)
        if success:
            app.temp_files['eda'] = temp_path
        eda_status = html.Div(message, style={'color': 'green' if success else 'red'})
    
    elif triggered_id == 'upload-pulse' and pulse_contents:
        temp_path, success, message = parse_contents(pulse_contents, pulse_filename)
        if success:
            app.temp_files['pulse'] = temp_path
        pulse_status = html.Div(message, style={'color': 'green' if success else 'red'})
    
    elif triggered_id == 'upload-temp' and temp_contents:
        temp_path, success, message = parse_contents(temp_contents, temp_filename)
        if success:
            app.temp_files['temp'] = temp_path
        temp_status = html.Div(message, style={'color': 'green' if success else 'red'})
    
    elif triggered_id == 'upload-accel' and accel_contents:
        temp_path, success, message = parse_contents(accel_contents, accel_filename)
        if success:
            app.temp_files['accel'] = temp_path
        accel_status = html.Div(message, style={'color': 'green' if success else 'red'})
    
    # Enable process button if all files are uploaded
    process_disabled = not all(app.temp_files.values())
    
    return eda_status, pulse_status, temp_status, accel_status, process_disabled

# Callback for window size selection buttons
@app.callback(
    Output('window-sizes', 'value'),
    [Input('select-all-btn', 'n_clicks'),
     Input('deselect-all-btn', 'n_clicks')],
    [State('window-sizes', 'options'),
     State('window-sizes', 'value')]
)
def update_window_sizes(select_clicks, deselect_clicks, options, current_value):
    if ctx.triggered_id == 'select-all-btn':
        return [option['value'] for option in options]
    elif ctx.triggered_id == 'deselect-all-btn':
        return []
    return current_value

# Callback to update threshold value display
@app.callback(
    Output('threshold-value', 'children'),
    [Input('threshold-slider', 'value')]
)
def update_threshold_value(value):
    return f"{value:.1f}"

# State Change Detection Algorithm
class StateChangeDetector:
    """
    Class for detecting psychological state changes from biomarker data.
    """
    
    def __init__(self, threshold: float = 2.0, window_sizes: list = None):
        """
        Initialize the state change detector.
        
        Args:
            threshold: Number of standard deviations to consider a change significant
            window_sizes: List of window sizes in minutes to analyze
        """
        self.threshold = threshold
        self.window_sizes = window_sizes or [3, 4, 5, 6, 8, 10, 15, 20, 30]
        
    def load_data(self, eda_file, pulse_rate_file, temperature_file, accelerometers_file):
        """
        Load data from CSV files.
        
        Args:
            eda_file: EDA data CSV file
            pulse_rate_file: Pulse rate data CSV file
            temperature_file: Temperature data CSV file
            accelerometers_file: Accelerometers data CSV file
            
        Returns:
            Dictionary containing processed dataframes for each biomarker
        """
        # Load data from CSV files
        eda_df = pd.read_csv(eda_file)
        pulse_rate_df = pd.read_csv(pulse_rate_file)
        temperature_df = pd.read_csv(temperature_file)
        accelerometers_df = pd.read_csv(accelerometers_file)
        
        # Process each dataframe
        self.eda_data = self._process_dataframe(eda_df, 'eda_scl_usiemens')
        self.pulse_rate_data = self._process_dataframe(pulse_rate_df, 'pulse_rate_bpm')
        self.temperature_data = self._process_dataframe(temperature_df, 'temperature_celsius')
        self.accelerometers_data = self._process_dataframe(accelerometers_df, 'accelerometers_std_g')
        
        return {
            'eda': self.eda_data,
            'pulse_rate': self.pulse_rate_data,
            'temperature': self.temperature_data,
            'accelerometers': self.accelerometers_data
        }
    
    def _process_dataframe(self, df: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """
        Process a dataframe to prepare it for analysis.
        
        Args:
            df: Input dataframe
            value_column: Name of the column containing the biomarker values
            
        Returns:
            Processed dataframe
        """
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp_iso'])
        
        # Filter out rows with missing values
        df = df[df['missing_value_reason'].isna()]
        
        # Select only necessary columns
        if value_column in df.columns:
            return df[['timestamp', value_column]].copy()
        else:
            # Return empty dataframe with correct columns if value column doesn't exist
            return pd.DataFrame(columns=['timestamp', value_column])
    
    def detect_state_changes(self) -> dict:
        """
        Detect state changes across all biomarkers and window sizes.
        
        Returns:
            Dictionary containing detected state changes and biomarker data
        """
        # Dictionary to store results for each biomarker and window size
        all_changes = {
            'eda': {},
            'pulse_rate': {},
            'temperature': {},
            'accelerometers': {}
        }
        
        # Detect changes for each biomarker and window size
        for biomarker, data in [
            ('eda', self.eda_data), 
            ('pulse_rate', self.pulse_rate_data),
            ('temperature', self.temperature_data),
            ('accelerometers', self.accelerometers_data)
        ]:
            if data.empty:
                continue
                
            value_column = data.columns[1]  # The second column contains the values
            
            for window_size in self.window_sizes:
                # Calculate rolling statistics
                window_changes = self._detect_changes_for_window(data, value_column, window_size)
                
                if window_changes:
                    all_changes[biomarker][window_size] = window_changes
        
        # Cross-validate changes across window sizes
        validated_changes = self._cross_validate_changes(all_changes)
        
        # Integrate changes across biomarkers
        integrated_changes = self._integrate_biomarker_changes(validated_changes)
        
        # Prepare data for visualization
        biomarker_data = self._prepare_biomarker_data()
        
        return {
            'state_changes': integrated_changes,
            'biomarker_data': biomarker_data
        }
    
    def _detect_changes_for_window(self, data: pd.DataFrame, value_column: str, window_size: int) -> list:
        """
        Detect significant changes for a specific window size.
        
        Args:
            data: Biomarker dataframe
            value_column: Name of the column containing the biomarker values
            window_size: Window size in minutes
            
        Returns:
            List of detected changes
        """
        # Convert window size from minutes to number of data points
        # Assuming data is sampled every minute
        window = window_size
        
        # Calculate rolling mean and standard deviation
        rolling_mean = data[value_column].rolling(window=window, min_periods=1).mean()
        rolling_std = data[value_column].rolling(window=window, min_periods=1).std()
        
        # Initialize list to store changes
        changes = []
        
        # Detect significant changes
        for i in range(window, len(data)):
            # Calculate z-score
            current_value = data[value_column].iloc[i]
            previous_mean = rolling_mean.iloc[i-1]
            previous_std = rolling_std.iloc[i-1]
            
            # Skip if standard deviation is zero or NaN
            if previous_std == 0 or np.isnan(previous_std):
                continue
                
            z_score = abs((current_value - previous_mean) / previous_std)
            
            # Check if change is significant
            if z_score > self.threshold:
                # Determine direction of change
                direction = 'increase' if current_value > previous_mean else 'decrease'
                
                # Add change to list
                changes.append({
                    'timestamp': data['timestamp'].iloc[i],
                    'value': current_value,
                    'previous_mean': previous_mean,
                    'z_score': z_score,
                    'direction': direction
                })
        
        return changes
    
    def _cross_validate_changes(self, all_changes: dict) -> dict:
        """
        Cross-validate changes across different window sizes.
        
        Args:
            all_changes: Dictionary containing changes for each biomarker and window size
            
        Returns:
            Dictionary containing validated changes for each biomarker
        """
        validated_changes = {}
        
        for biomarker, window_changes in all_changes.items():
            if not window_changes:
                continue
                
            # Get all unique timestamps across all window sizes
            all_timestamps = set()
            for window_size, changes in window_changes.items():
                for change in changes:
                    all_timestamps.add(change['timestamp'])
            
            # Count occurrences of each timestamp across window sizes
            timestamp_counts = {}
            for timestamp in all_timestamps:
                count = 0
                for window_size, changes in window_changes.items():
                    for change in changes:
                        if change['timestamp'] == timestamp:
                            count += 1
                            break
                timestamp_counts[timestamp] = count
            
            # Filter timestamps that appear in at least 2 window sizes
            validated_timestamps = [ts for ts, count in timestamp_counts.items() if count >= 2]
            
            # Get the details of validated changes
            validated_changes[biomarker] = []
            for timestamp in validated_timestamps:
                # Find the change with the highest z-score for this timestamp
                best_change = None
                best_z_score = 0
                
                for window_size, changes in window_changes.items():
                    for change in changes:
                        if change['timestamp'] == timestamp and change['z_score'] > best_z_score:
                            best_change = change
                            best_z_score = change['z_score']
                
                if best_change:
                    validated_changes[biomarker].append(best_change)
        
        return validated_changes
    
    def _integrate_biomarker_changes(self, validated_changes: dict) -> list:
        """
        Integrate changes across different biomarkers.
        
        Args:
            validated_changes: Dictionary containing validated changes for each biomarker
            
        Returns:
            List of integrated state changes
        """
        # Get all unique timestamps across all biomarkers
        all_timestamps = set()
        for biomarker, changes in validated_changes.items():
            for change in changes:
                all_timestamps.add(change['timestamp'])
        
        # Group changes by timestamp
        grouped_changes = {}
        for timestamp in all_timestamps:
            grouped_changes[timestamp] = {}
            
            for biomarker, changes in validated_changes.items():
                for change in changes:
                    if change['timestamp'] == timestamp:
                        grouped_changes[timestamp][biomarker] = change
        
        # Filter timestamps with changes in at least 2 biomarkers
        integrated_changes = []
        for timestamp, biomarker_changes in grouped_changes.items():
            if len(biomarker_changes) >= 2:
                # Calculate average z-score as confidence
                z_scores = [change['z_score'] for change in biomarker_changes.values()]
                avg_z_score = sum(z_scores) / len(z_scores)
                
                # Determine probable state transition
                probable_transition = self._determine_state_transition(biomarker_changes)
                
                # Format biomarker changes for display
                formatted_changes = {}
                for biomarker, change in biomarker_changes.items():
                    formatted_changes[biomarker] = {
                        'direction': change['direction'],
                        'from': change['previous_mean'],
                        'to': change['value']
                    }
                
                # Add integrated change to list
                integrated_changes.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'biomarker_changes': formatted_changes,
                    'probable_transition': probable_transition,
                    'confidence': avg_z_score
                })
        
        # Sort changes by timestamp
        integrated_changes.sort(key=lambda x: x['timestamp'])
        
        return integrated_changes
    
    def _determine_state_transition(self, biomarker_changes: dict) -> str:
        """
        Determine the probable state transition based on biomarker changes.
        
        Args:
            biomarker_changes: Dictionary containing changes for each biomarker
            
        Returns:
            String describing the probable state transition
        """
        # Check for patterns indicating specific transitions
        
        # Anxiety/Irritation pattern: Increased EDA, increased heart rate, decreased movement
        anxiety_pattern = (
            biomarker_changes.get('eda', {}).get('direction') == 'increase' and
            biomarker_changes.get('pulse_rate', {}).get('direction') == 'increase' and
            (
                'accelerometers' not in biomarker_changes or
                biomarker_changes.get('accelerometers', {}).get('direction') == 'decrease'
            )
        )
        
        if anxiety_pattern:
            return "Neutral → Anxious/Irritated"
        
        # Calm/Relaxation pattern: Decreased EDA, decreased heart rate
        calm_pattern = (
            biomarker_changes.get('eda', {}).get('direction') == 'decrease' and
            biomarker_changes.get('pulse_rate', {}).get('direction') == 'decrease'
        )
        
        if calm_pattern:
            return "Anxious/Irritated → Neutral"
        
        # Excitement pattern: Increased EDA, increased heart rate, increased movement
        excitement_pattern = (
            biomarker_changes.get('eda', {}).get('direction') == 'increase' and
            biomarker_changes.get('pulse_rate', {}).get('direction') == 'increase' and
            biomarker_changes.get('accelerometers', {}).get('direction') == 'increase'
        )
        
        if excitement_pattern:
            return "Neutral → Excited"
        
        # Default: Unclassified state change
        return "Unclassified State Change"
    
    def _prepare_biomarker_data(self) -> dict:
        """
        Prepare biomarker data for visualization.
        
        Returns:
            Dictionary containing formatted biomarker data
        """
        biomarker_data = {}
        
        # Process each biomarker
        for biomarker, data in [
            ('eda', self.eda_data), 
            ('pulse_rate', self.pulse_rate_data),
            ('temperature', self.temperature_data),
            ('accelerometers', self.accelerometers_data)
        ]:
            if data.empty:
                continue
                
            value_column = data.columns[1]  # The second column contains the values
            
            # Format timestamps and values
            timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in data['timestamp']]
            values = data[value_column].tolist()
            
            biomarker_data[biomarker] = {
                'timestamps': timestamps,
                'values': values
            }
        
        return biomarker_data

# Callback for processing data
@app.callback(
    [Output('loading-output', 'children'),
     Output('results-container', 'style'),
     Output('state-changes-table', 'children'),
     Output('eda-graph', 'figure'),
     Output('pulse-graph', 'figure'),
     Output('temp-graph', 'figure'),
     Output('accel-graph', 'figure')],
    [Input('process-btn', 'n_clicks')],
    [State('threshold-slider', 'value'),
     State('window-sizes', 'value')]
)
def process_data(n_clicks, threshold, window_sizes):
    if n_clicks is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    # Check if all files are uploaded
    if not all(app.temp_files.values()):
        return "Please upload all required files.", no_update, no_update, no_update, no_update, no_update, no_update
    
    # Initialize state change detector
    detector = StateChangeDetector(threshold=threshold, window_sizes=window_sizes)
    
    # Load data
    detector.load_data(
        app.temp_files['eda'],
        app.temp_files['pulse'],
        app.temp_files['temp'],
        app.temp_files['accel']
    )
    
    # Detect state changes
    results = detector.detect_state_changes()
    
    # Store processed data
    app.processed_data = results
    
    # Create state changes table
    if not results['state_changes']:
        state_changes_table = html.Div("No significant state changes detected with the current threshold.", 
                                      className="alert alert-info")
    else:
        state_changes_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Timestamp"),
                html.Th("Biomarker Changes"),
                html.Th("Probable Transition"),
                html.Th("Confidence")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(change['timestamp']),
                    html.Td(html.Div([
                        html.Div([
                            f"{biomarker}: {'↑' if details['direction'] == 'increase' else '↓'} "
                            f"{details['from']:.2f} → {details['to']:.2f}"
                        ]) for biomarker, details in change['biomarker_changes'].items()
                    ])),
                    html.Td(change['probable_transition']),
                    html.Td(f"{change['confidence']:.2f}")
                ]) for change in results['state_changes']
            ])
        ], striped=True, bordered=True, hover=True)
    
    # Create biomarker graphs
    biomarker_graphs = {}
    
    for biomarker in ['eda', 'pulse_rate', 'temperature', 'accelerometers']:
        if biomarker not in results['biomarker_data']:
            biomarker_graphs[biomarker] = go.Figure()
            continue
        
        data = results['biomarker_data'][biomarker]
        
        fig = go.Figure()
        
        # Add main data line
        fig.add_trace(go.Scatter(
            x=data['timestamps'],
            y=data['values'],
            mode='lines',
            name=biomarker,
            line=dict(color=get_color_for_biomarker(biomarker))
        ))
        
        # Add markers for state changes
        for change in results['state_changes']:
            if biomarker in change['biomarker_changes']:
                # Find the index of this timestamp
                try:
                    idx = data['timestamps'].index(change['timestamp'])
                    
                    fig.add_trace(go.Scatter(
                        x=[change['timestamp']],
                        y=[data['values'][idx]],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='circle-open',
                            line=dict(width=2)
                        ),
                        name=f"State Change: {change['probable_transition']}",
                        hoverinfo='text',
                        hovertext=f"State Change: {change['probable_transition']}<br>"
                                 f"Confidence: {change['confidence']:.2f}<br>"
                                 f"Value: {data['values'][idx]:.2f}"
                    ))
                except ValueError:
                    continue
        
        # Update layout
        fig.update_layout(
            title=f"{biomarker.replace('_', ' ').title()} Data",
            xaxis_title="Time",
            yaxis_title=get_biomarker_unit(biomarker),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        biomarker_graphs[biomarker] = fig
    
    return "", {"display": "block"}, state_changes_table, biomarker_graphs['eda'], biomarker_graphs['pulse_rate'], biomarker_graphs['temperature'], biomarker_graphs['accelerometers']

def get_color_for_biomarker(biomarker):
    colors = {
        'eda': 'rgb(75, 192, 192)',
        'pulse_rate': 'rgb(255, 99, 132)',
        'temperature': 'rgb(255, 159, 64)',
        'accelerometers': 'rgb(54, 162, 235)'
    }
    return colors.get(biomarker, 'rgb(128, 128, 128)')

def get_biomarker_unit(biomarker):
    units = {
        'eda': 'μS',
        'pulse_rate': 'BPM',
        'temperature': '°C',
        'accelerometers': 'g'
    }
    return units.get(biomarker, '')

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)
