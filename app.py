import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import base64
import io
from scipy.signal import find_peaks

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose server for Gunicorn

# Define the app layout
app.layout = html.Div([
    dbc.Container([
        html.H1("State Change Detection for Empatica Wristband", className="text-center my-4"),
        html.P("Upload your Empatica wristband data files to detect psychological state changes", className="text-center mb-5"),
        
        # File upload section
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
                        html.Label("Window Overlap (%): ", className="mt-3"),
                        html.Span(id="overlap-value", children="0"),
                        dcc.Slider(
                            id='overlap-slider',
                            min=0,
                            max=75,
                            step=5,
                            value=0,
                            marks={i: str(i) for i in range(0, 76, 15)}
                        )
                    ], md=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Peak Detection Sensitivity: ", className="mt-3"),
                        html.Span(id="sensitivity-value", children="0.7"),
                        dcc.Slider(
                            id='sensitivity-slider',
                            min=0.3,
                            max=1.0,
                            step=0.05,
                            value=0.7,
                            marks={i/10: str(i/10) for i in range(3, 11, 1)}
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
            
            html.Div(id="biomarker-tabs")
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
            # Save to temp file
            temp_file = io.StringIO(decoded.decode('utf-8'))
            return temp_file, f"Uploaded: {filename}"
        else:
            return None, f"Error: {filename} is not a CSV file"
    except Exception as e:
        return None, f"Error processing {filename}: {str(e)}"

# Callback for EDA file upload
@app.callback(
    [Output('eda-upload-status', 'children'),
     Output('process-btn', 'disabled', allow_duplicate=True)],
    [Input('upload-eda', 'contents')],
    [State('upload-eda', 'filename'),
     State('upload-pulse', 'contents'),
     State('upload-temp', 'contents'),
     State('upload-accel', 'contents')],
    prevent_initial_call=True
)
def update_eda_upload(contents, filename, pulse_contents, temp_contents, accel_contents):
    if contents is None:
        return "No file uploaded.", True
    
    temp_file, message = parse_contents(contents, filename)
    app.temp_files['eda'] = temp_file
    
    # Enable process button if all files are uploaded
    all_files_uploaded = all([
        app.temp_files['eda'] is not None,
        app.temp_files['pulse'] is not None,
        app.temp_files['temp'] is not None,
        app.temp_files['accel'] is not None
    ])
    
    return message, not all_files_uploaded

# Callback for pulse rate file upload
@app.callback(
    [Output('pulse-upload-status', 'children'),
     Output('process-btn', 'disabled', allow_duplicate=True)],
    [Input('upload-pulse', 'contents')],
    [State('upload-pulse', 'filename'),
     State('upload-eda', 'contents'),
     State('upload-temp', 'contents'),
     State('upload-accel', 'contents')],
    prevent_initial_call=True
)
def update_pulse_upload(contents, filename, eda_contents, temp_contents, accel_contents):
    if contents is None:
        return "No file uploaded.", True
    
    temp_file, message = parse_contents(contents, filename)
    app.temp_files['pulse'] = temp_file
    
    # Enable process button if all files are uploaded
    all_files_uploaded = all([
        app.temp_files['eda'] is not None,
        app.temp_files['pulse'] is not None,
        app.temp_files['temp'] is not None,
        app.temp_files['accel'] is not None
    ])
    
    return message, not all_files_uploaded

# Callback for temperature file upload
@app.callback(
    [Output('temp-upload-status', 'children'),
     Output('process-btn', 'disabled', allow_duplicate=True)],
    [Input('upload-temp', 'contents')],
    [State('upload-temp', 'filename'),
     State('upload-eda', 'contents'),
     State('upload-pulse', 'contents'),
     State('upload-accel', 'contents')],
    prevent_initial_call=True
)
def update_temp_upload(contents, filename, eda_contents, pulse_contents, accel_contents):
    if contents is None:
        return "No file uploaded.", True
    
    temp_file, message = parse_contents(contents, filename)
    app.temp_files['temp'] = temp_file
    
    # Enable process button if all files are uploaded
    all_files_uploaded = all([
        app.temp_files['eda'] is not None,
        app.temp_files['pulse'] is not None,
        app.temp_files['temp'] is not None,
        app.temp_files['accel'] is not None
    ])
    
    return message, not all_files_uploaded

# Callback for accelerometers file upload
@app.callback(
    [Output('accel-upload-status', 'children'),
     Output('process-btn', 'disabled', allow_duplicate=True)],
    [Input('upload-accel', 'contents')],
    [State('upload-accel', 'filename'),
     State('upload-eda', 'contents'),
     State('upload-pulse', 'contents'),
     State('upload-temp', 'contents')],
    prevent_initial_call=True
)
def update_accel_upload(contents, filename, eda_contents, pulse_contents, temp_contents):
    if contents is None:
        return "No file uploaded.", True
    
    temp_file, message = parse_contents(contents, filename)
    app.temp_files['accel'] = temp_file
    
    # Enable process button if all files are uploaded
    all_files_uploaded = all([
        app.temp_files['eda'] is not None,
        app.temp_files['pulse'] is not None,
        app.temp_files['temp'] is not None,
        app.temp_files['accel'] is not None
    ])
    
    return message, not all_files_uploaded

# Callback for window size selection
@app.callback(
    Output('window-sizes', 'value'),
    [Input('select-all-btn', 'n_clicks'),
     Input('deselect-all-btn', 'n_clicks')],
    [State('window-sizes', 'options'),
     State('window-sizes', 'value')],
    prevent_initial_call=True
)
def update_window_sizes(select_clicks, deselect_clicks, options, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value
    
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

# Callback to update overlap value display
@app.callback(
    Output('overlap-value', 'children'),
    [Input('overlap-slider', 'value')]
)
def update_overlap_value(value):
    return f"{value}"

# Callback to update sensitivity value display
@app.callback(
    Output('sensitivity-value', 'children'),
    [Input('sensitivity-slider', 'value')]
)
def update_sensitivity_value(value):
    return f"{value:.2f}"

# State Change Detection Algorithm
class StateChangeDetector:
    """
    Algorithm for detecting psychological state changes from Empatica wristband data.
    """
    
    def __init__(self, threshold=2.0, window_sizes=None, overlap_percent=0, sensitivity=0.7):
        """
        Initialize the state change detector.
        
        Args:
            threshold: Number of standard deviations to consider a change significant
            window_sizes: List of window sizes in minutes to analyze
            overlap_percent: Percentage of overlap between consecutive windows (0-100)
            sensitivity: Sensitivity factor for peak detection (0.3-1.0)
        """
        self.threshold = threshold
        self.window_sizes = window_sizes or [3, 4, 5, 6, 8, 10, 15, 20, 30]
        self.overlap_percent = max(0, min(overlap_percent, 100))  # Ensure between 0-100%
        self.sensitivity = max(0.3, min(sensitivity, 1.0))  # Ensure between 0.3-1.0
        
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
                window_changes = self._detect_changes_for_window(data, value_column, window_size, biomarker)
                
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
    
    def _detect_changes_for_window(self, data: pd.DataFrame, value_column: str, window_size: int, biomarker: str) -> list:
        """
        Detect significant changes for a specific window size.
        
        Args:
            data: Biomarker dataframe
            value_column: Name of the column containing the biomarker values
            window_size: Window size in minutes
            biomarker: Name of the biomarker being analyzed
            
        Returns:
            List of detected changes
        """
        # Convert window size from minutes to number of data points
        # Assuming data is sampled every minute
        window = window_size
        
        # Calculate step size based on overlap percentage
        step_size = max(1, int(window * (1 - self.overlap_percent / 100)))
        
        # Initialize list to store changes
        changes = []
        
        # Detect significant changes with overlapping windows
        for i in range(window, len(data), step_size):
            # Get window start index
            start_idx = i - window
            
            # Calculate window statistics
            window_data = data[value_column].iloc[start_idx:i]
            window_mean = window_data.mean()
            window_std = window_data.std()
            
            # Skip if standard deviation is zero or NaN
            if window_std == 0 or np.isnan(window_std):
                continue
            
            # Get current value (first point after the window)
            if i < len(data):
                current_value = data[value_column].iloc[i]
                
                # Calculate z-score
                z_score = abs((current_value - window_mean) / window_std)
                
                # Check if change is significant
                if z_score > self.threshold:
                    # Determine direction of change
                    direction = 'increase' if current_value > window_mean else 'decrease'
                    
                    # Add change to list
                    changes.append({
                        'timestamp': data['timestamp'].iloc[i],
                        'value': current_value,
                        'previous_mean': window_mean,
                        'previous_std': window_std,
                        'z_score': z_score,
                        'direction': direction,
                        'window_size': window_size,
                        'window_start': data['timestamp'].iloc[start_idx],
                        'window_end': data['timestamp'].iloc[i-1],
                        'overlap_percent': self.overlap_percent,
                        'detection_type': 'window_change'
                    })
        
        # Add peak detection for significant local maxima and minima
        self._add_peak_detections(data, value_column, window_size, changes, biomarker)
        
        return changes
    
    def _add_peak_detections(self, data: pd.DataFrame, value_column: str, window_size: int, changes: list, biomarker: str):
        """
        Add peak detections based on local maxima and minima.
        
        Args:
            data: Input dataframe
            value_column: Name of the column containing the biomarker values
            window_size: Window size in minutes
            changes: List of existing changes to append to
            biomarker: Name of the biomarker being analyzed
        """
        if len(data) < window_size * 2:
            return  # Not enough data for peak detection
            
        # Calculate rolling mean and standard deviation for the entire dataset
        # This helps establish a baseline for peak detection
        rolling_mean = data[value_column].rolling(window=window_size, center=True).mean()
        rolling_std = data[value_column].rolling(window=window_size, center=True).std()
        
        # Fill NaN values at the edges
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        # Adjust threshold based on sensitivity factor
        adjusted_threshold = self.threshold * self.sensitivity
        
        # Find peaks (local maxima)
        # We use scipy.signal.find_peaks which identifies local maxima in the data
        values = data[value_column].values
        
        # For peaks above the mean
        # Height parameter ensures we only detect peaks that exceed the threshold
        # Distance parameter ensures peaks are not too close to each other
        peaks, peak_properties = find_peaks(
            values, 
            height=rolling_mean + (rolling_std * adjusted_threshold),
            distance=max(3, window_size // 3)  # Minimum distance between peaks
        )
        
        # For troughs below the mean
        # We invert the signal to find troughs as peaks
        troughs, trough_properties = find_peaks(
            -values, 
            height=-(rolling_mean - (rolling_std * adjusted_threshold)),
            distance=max(3, window_size // 3)
        )
        
        # Process peaks
        for peak_idx in peaks:
            if peak_idx >= len(data):
                continue
                
            peak_value = data.iloc[peak_idx][value_column]
            peak_time = data.iloc[peak_idx]['timestamp']
            
            # Calculate local z-score relative to surrounding data
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(data), peak_idx + window_size)
            local_data = data.iloc[start_idx:end_idx]
            local_mean = local_data[value_column].mean()
            local_std = local_data[value_column].std()
            
            if local_std == 0 or np.isnan(local_std):
                continue
                
            z_score = abs(peak_value - local_mean) / local_std
            
            # Only add if z-score exceeds threshold and not too close to existing changes
            if z_score > adjusted_threshold:
                # Check if this peak is already covered by an existing change
                is_duplicate = False
                for change in changes:
                    time_diff = abs((peak_time - change['timestamp']).total_seconds())
                    if time_diff < window_size * 60 / 2:  # Half window size in seconds
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    changes.append({
                        'timestamp': peak_time,
                        'value': peak_value,
                        'previous_mean': local_mean,
                        'previous_std': local_std,
                        'z_score': z_score,
                        'direction': 'increase',
                        'window_size': window_size,
                        'window_start': local_data.iloc[0]['timestamp'],
                        'window_end': local_data.iloc[-1]['timestamp'],
                        'overlap_percent': self.overlap_percent,
                        'detection_type': 'peak',
                        'biomarker': biomarker
                    })
        
        # Process troughs
        for trough_idx in troughs:
            if trough_idx >= len(data):
                continue
                
            trough_value = data.iloc[trough_idx][value_column]
            trough_time = data.iloc[trough_idx]['timestamp']
            
            # Calculate local z-score
            start_idx = max(0, trough_idx - window_size)
            end_idx = min(len(data), trough_idx + window_size)
            local_data = data.iloc[start_idx:end_idx]
            local_mean = local_data[value_column].mean()
            local_std = local_data[value_column].std()
            
            if local_std == 0 or np.isnan(local_std):
                continue
                
            z_score = abs(trough_value - local_mean) / local_std
            
            # Only add if z-score exceeds threshold and not too close to existing changes
            if z_score > adjusted_threshold:
                # Check if this trough is already covered by an existing change
                is_duplicate = False
                for change in changes:
                    time_diff = abs((trough_time - change['timestamp']).total_seconds())
                    if time_diff < window_size * 60 / 2:  # Half window size in seconds
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    changes.append({
                        'timestamp': trough_time,
                        'value': trough_value,
                        'previous_mean': local_mean,
                        'previous_std': local_std,
                        'z_score': z_score,
                        'direction': 'decrease',
                        'window_size': window_size,
                        'window_start': local_data.iloc[0]['timestamp'],
                        'window_end': local_data.iloc[-1]['timestamp'],
                        'overlap_percent': self.overlap_percent,
                        'detection_type': 'trough',
                        'biomarker': biomarker
                    })
    
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
            
            # Group changes by timestamp
            grouped_changes = {}
            for timestamp in all_timestamps:
                # Find changes across window sizes that match this timestamp
                matching_changes = []
                for window_size, changes in window_changes.items():
                    for change in changes:
                        # Check if timestamps are close (within 1 minute)
                        time_diff = abs((timestamp - change['timestamp']).total_seconds())
                        if time_diff <= 60:  # 1 minute in seconds
                            matching_changes.append(change)
                
                # Only keep timestamp if it appears in at least 2 window sizes
                # or if it's a peak/trough detection (which are more reliable)
                window_sizes_count = len(set(change['window_size'] for change in matching_changes))
                has_peak_detection = any(change.get('detection_type') in ['peak', 'trough'] for change in matching_changes)
                
                if window_sizes_count >= 2 or has_peak_detection:
                    # Use the change with the highest z-score
                    best_change = max(matching_changes, key=lambda x: x['z_score'])
                    grouped_changes[timestamp] = best_change
            
            validated_changes[biomarker] = list(grouped_changes.values())
        
        return validated_changes
    
    def _integrate_biomarker_changes(self, validated_changes: dict) -> list:
        """
        Integrate changes across biomarkers.
        
        Args:
            validated_changes: Dictionary containing validated changes for each biomarker
            
        Returns:
            List of integrated changes
        """
        # Flatten all changes across biomarkers
        all_changes = []
        for biomarker, changes in validated_changes.items():
            for change in changes:
                all_changes.append({
                    'biomarker': biomarker,
                    'timestamp': change['timestamp'],
                    'z_score': change['z_score'],
                    'direction': change['direction'],
                    'window_size': change['window_size'],
                    'overlap_percent': change['overlap_percent'],
                    'window_start': change['window_start'],
                    'window_end': change['window_end'],
                    'detection_type': change.get('detection_type', 'window_change')
                })
        
        # Sort changes by timestamp
        all_changes.sort(key=lambda x: x['timestamp'])
        
        # Group changes by timestamp (within a 2-minute window)
        grouped_changes = {}
        for change in all_changes:
            # Find if there's a close timestamp already in the groups
            found_match = False
            for timestamp in list(grouped_changes.keys()):
                time_diff = abs((timestamp - change['timestamp']).total_seconds())
                if time_diff <= 120:  # 2 minutes in seconds
                    grouped_changes[timestamp][change['biomarker']] = change
                    found_match = True
                    break
            
            if not found_match:
                grouped_changes[change['timestamp']] = {change['biomarker']: change}
        
        # Filter timestamps with changes in at least 2 biomarkers
        # or with high-confidence peak detections
        integrated_changes = []
        for timestamp, biomarker_changes in grouped_changes.items():
            # Check if we have multiple biomarkers or high-confidence peak detections
            has_multiple_biomarkers = len(biomarker_changes) >= 2
            has_peak_detection = any(
                change.get('detection_type') in ['peak', 'trough'] and change['z_score'] > self.threshold * 1.5
                for change in biomarker_changes.values()
            )
            
            if has_multiple_biomarkers or has_peak_detection:
                # Calculate average z-score
                z_scores = [change['z_score'] for change in biomarker_changes.values()]
                avg_z_score = sum(z_scores) / len(z_scores)
                
                # Calculate confidence percentage (0-100%)
                # Using a sigmoid function to map z-scores to 0-100% range
                confidence_pct = 100 * (1 / (1 + np.exp(-avg_z_score + self.threshold)))
                
                # Get window sizes and overlap info
                window_sizes = {biomarker: change.get('window_size', 'N/A') for biomarker, change in biomarker_changes.items()}
                overlap_percent = next(iter(biomarker_changes.values())).get('overlap_percent', 0)
                
                # Determine probable state transition
                probable_transition = self._determine_state_transition(biomarker_changes)
                
                # Format biomarker changes for display
                formatted_changes = {}
                for biomarker, change in biomarker_changes.items():
                    formatted_changes[biomarker] = {
                        'direction': change['direction'],
                        'z_score': change['z_score'],
                        'window_size': change['window_size'],
                        'window_start': change.get('window_start', None),
                        'window_end': change.get('window_end', None),
                        'detection_type': change.get('detection_type', 'window_change')
                    }
                
                # Add integrated change to list
                integrated_changes.append({
                    'timestamp': timestamp,
                    'biomarkers': formatted_changes,
                    'z_score': avg_z_score,
                    'confidence_pct': confidence_pct,
                    'probable_transition': probable_transition,
                    'window_sizes': window_sizes,
                    'overlap_percent': overlap_percent
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
        # Check for relaxation pattern
        relaxation_pattern = (
            biomarker_changes.get('eda', {}).get('direction') == 'decrease' and
            biomarker_changes.get('pulse_rate', {}).get('direction') == 'decrease'
        )
        
        if relaxation_pattern:
            return "Aroused → Relaxed"
        
        # Check for stress/arousal pattern
        stress_pattern = (
            biomarker_changes.get('eda', {}).get('direction') == 'increase' and
            biomarker_changes.get('pulse_rate', {}).get('direction') == 'increase'
        )
        
        if stress_pattern:
            return "Relaxed → Aroused"
        
        # Check for excitement pattern
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
     Output('biomarker-tabs', 'children')],
    [Input('process-btn', 'n_clicks')],
    [State('threshold-slider', 'value'),
     State('window-sizes', 'value'),
     State('overlap-slider', 'value'),
     State('sensitivity-slider', 'value')]
)
def process_data(n_clicks, threshold, window_sizes, overlap_percent, sensitivity):
    if n_clicks is None:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    
    # Check if all files are uploaded
    if not all(app.temp_files.values()):
        return "Please upload all required files.", {'display': 'none'}, None, None
    
    # Initialize state change detector with all parameters
    detector = StateChangeDetector(
        threshold=threshold, 
        window_sizes=window_sizes, 
        overlap_percent=overlap_percent,
        sensitivity=sensitivity
    )
    
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
        state_changes_table = html.Div("No significant state changes detected with the current parameters.", 
                                      className="alert alert-info")
    else:
        state_changes_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Timestamp"),
                html.Th("Biomarker Changes"),
                html.Th("Detection Type"),
                html.Th("Window Sizes (min)"),
                html.Th("Overlap (%)"),
                html.Th("Probable Transition"),
                html.Th("Z-Score"),
                html.Th("Confidence (%)")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(change['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(change['timestamp'], pd.Timestamp) else str(change['timestamp'])),
                    html.Td(html.Div([
                        html.Div([
                            f"{biomarker}: {'↑' if details['direction'] == 'increase' else '↓'} "
                            f"z-score: {details['z_score']:.2f}"
                        ]) for biomarker, details in change['biomarkers'].items()
                    ])),
                    html.Td(html.Div([
                        html.Div([
                            f"{details.get('detection_type', 'window_change')}"
                        ]) for biomarker, details in change['biomarkers'].items()
                    ])),
                    html.Td(html.Div([
                        html.Div([
                            f"{biomarker}: {size}"
                        ]) for biomarker, size in change['window_sizes'].items()
                    ])),
                    html.Td(f"{change['overlap_percent']}%"),
                    html.Td(change['probable_transition']),
                    html.Td(f"{change['z_score']:.2f}"),
                    html.Td(f"{change['confidence_pct']:.1f}%")
                ]) for change in results['state_changes']
            ])
        ], striped=True, bordered=True, hover=True)
    
    # Create biomarker tabs
    tabs = []
    
    # Create a single figure with subplots for all biomarkers
    biomarker_names = list(results['biomarker_data'].keys())
    if biomarker_names:
        # Create a figure with subplots (one row per biomarker)
        fig = go.Figure()
        subplot_height = 250  # Height per subplot in pixels
        total_height = subplot_height * len(biomarker_names)
        
        # Calculate y-axis domains for each subplot
        domains = []
        for i in range(len(biomarker_names)):
            domain_start = 1 - ((i + 1) / len(biomarker_names))
            domain_end = 1 - (i / len(biomarker_names))
            domains.append([domain_start, domain_end])
        
        # Add traces for each biomarker
        for i, biomarker in enumerate(biomarker_names):
            data = results['biomarker_data'][biomarker]
            if not data:
                continue
            
            # Add main data line
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['values'],
                mode='lines',
                name=biomarker,
                line=dict(color=get_color_for_biomarker(biomarker)),
                yaxis=f'y{i+1}'
            ))
            
            # Add markers for state changes
            for change in results['state_changes']:
                if biomarker in change['biomarkers']:
                    # Find the index of this timestamp
                    try:
                        timestamp_str = change['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(change['timestamp'], pd.Timestamp) else str(change['timestamp'])
                        if timestamp_str in data['timestamps']:
                            idx = data['timestamps'].index(timestamp_str)
                            
                            # Get window information
                            window_size = change['window_sizes'].get(biomarker, 'N/A')
                            window_start = change['biomarkers'][biomarker].get('window_start', None)
                            window_end = change['biomarkers'][biomarker].get('window_end', None)
                            detection_type = change['biomarkers'][biomarker].get('detection_type', 'window_change')
                            
                            # Format window info for hover text
                            window_info = ""
                            if window_start and window_end:
                                window_start_str = window_start.strftime('%H:%M:%S') if isinstance(window_start, pd.Timestamp) else str(window_start)
                                window_end_str = window_end.strftime('%H:%M:%S') if isinstance(window_end, pd.Timestamp) else str(window_end)
                                window_info = f"<br>Window: {window_start_str} to {window_end_str}<br>Window Size: {window_size} min<br>Overlap: {change['overlap_percent']}%"
                            
                            # Use different marker symbols based on detection type
                            marker_symbol = 'circle-open'
                            if detection_type == 'peak':
                                marker_symbol = 'triangle-up'
                            elif detection_type == 'trough':
                                marker_symbol = 'triangle-down'
                            
                            fig.add_trace(go.Scatter(
                                x=[timestamp_str],
                                y=[data['values'][idx]],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color='red',
                                    symbol=marker_symbol,
                                    line=dict(width=2)
                                ),
                                name=f"{biomarker} Change",
                                hoverinfo='text',
                                hovertext=f"State Change: {change['probable_transition']}<br>"
                                         f"Detection Type: {detection_type}<br>"
                                         f"Z-Score: {change['z_score']:.2f}<br>"
                                         f"Confidence: {change['confidence_pct']:.1f}%"
                                         f"{window_info}",
                                yaxis=f'y{i+1}'
                            ))
                            
                            # Add shaded area for the window if available
                            if window_start and window_end:
                                window_start_str = window_start.strftime('%Y-%m-%d %H:%M:%S') if isinstance(window_start, pd.Timestamp) else str(window_start)
                                window_end_str = window_end.strftime('%Y-%m-%d %H:%M:%S') if isinstance(window_end, pd.Timestamp) else str(window_end)
                                
                                # Find indices for window start and end
                                if window_start_str in data['timestamps'] and window_end_str in data['timestamps']:
                                    start_idx = data['timestamps'].index(window_start_str)
                                    end_idx = data['timestamps'].index(window_end_str)
                                    
                                    # Add shaded area
                                    fig.add_trace(go.Scatter(
                                        x=data['timestamps'][start_idx:end_idx+1],
                                        y=data['values'][start_idx:end_idx+1],
                                        fill='tozeroy',
                                        fillcolor='rgba(0, 255, 0, 0.2)',
                                        line=dict(color='rgba(0, 255, 0, 0.5)'),
                                        name=f"Window",
                                        hoverinfo='skip',
                                        yaxis=f'y{i+1}'
                                    ))
                    except (ValueError, IndexError):
                        continue
            
            # Create y-axis for this biomarker
            fig.update_layout(**{
                f'yaxis{i+1}': dict(
                    title=f"{biomarker.replace('_', ' ').title()} ({get_biomarker_unit(biomarker)})",
                    domain=domains[i],
                    titlefont=dict(color=get_color_for_biomarker(biomarker)),
                    tickfont=dict(color=get_color_for_biomarker(biomarker)),
                    gridcolor='rgba(0,0,0,0.1)'
                )
            })
        
        # Update layout
        fig.update_layout(
            title="Physiological Signals with Detected State Changes",
            xaxis=dict(
                title="Time",
                domain=[0, 1]
            ),
            height=total_height + 150,  # Add extra space for title and x-axis
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="closest",
            showlegend=False
        )
        
        # Create graph component
        graph = dcc.Graph(figure=fig, style={'height': f'{total_height + 150}px'})
        
        return "", {"display": "block"}, state_changes_table, graph
    else:
        return "", {"display": "block"}, state_changes_table, html.Div("No biomarker data available.")

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
