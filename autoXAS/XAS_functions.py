# #%% Imports

# # Packages for handling time
# from datetime import datetime
# import time
# # Packages for math
# import numpy as np
# # Packages for typing
# from typing import Any, Union
# # Packages for handling data
# import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
# from pathlib import Path
# # from tqdm.auto import tqdm # Use as standard. If progress bar is not rendering use normal tqdm below.
# from tqdm import tqdm
# # Packages for watching directories
# from watchdog.observers import Observer
# from watchdog.events import PatternMatchingEventHandler, FileSystemEventHandler
# # Packages for managing notebook and IPython features
# from IPython.display import clear_output
# # Packages for fitting
# from lmfit import Parameters, fit_report, minimize
# from lmfit.minimizer import MinimizerResult
# # Packages for plotting
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# import seaborn as sns
# sns.set_theme()
# import plotly_express as px
# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = 'notebook'
# #XAS specfific packages
# from larch.xray import xray_edge
# from larch.xafs import pre_edge, find_e0
# from larch import Group

# #%% Data loader functions

# def load_xas_data(
#     folder_path: str, 
#     synchrotron: str, 
#     keep_incomplete: bool=False,
#     file_selection_condition: Union[str,list[str]]='',
#     negated_condition: bool=False,
#     verbose: bool=True,
# ) -> pd.DataFrame:
#     """------------------------------------------------------------------
#     Load XAS data files (.dat extension) from a folder.

#     Args:
#         folder_path (str): Filepath to the folder containing the .dat files.
#         synchrotron (str): Name of the synchrotron the data was measured at.
#         keep_incomplete (optional, bool): Whether to keep incomplete measurements or not. Defaults to False.
#         file_selection_condition (optional, Union[str, list[str]]): Substring present in filenames to either load or ignore. Defaults to ' '.
#         negated_condition (optional, bool): Whether to load or ignore files with the given substring in the filename. Defaults to False.
#         verbose (optional, bool): Whether to print other things than progress bars. Defaults to True.

#     Returns:
#         pd.DataFrame: The raw data from the .dat files.
#     """    
#     # List of the synchrotrons which the function can load data from
#     implemented = ['ESRF', 'BALDER', 'BALDER_2']
#     # Check the data format can be handled
#     assert synchrotron in implemented, f'Loading of data from {synchrotron} is not implemented. The implemented synchrotrons are:\n\t{implemented}\n\nIf you want the loading of data from a specific facility implemented contact me at ufj@chem.ku.dk or submit a request at [github link].'
#     if synchrotron in ['ESRF', 'BALDER', 'BALDER_2']:
#         # Reads all the correct files
#         if type(file_selection_condition) == list:
#             if negated_condition:
#                 filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring not in path.stem for substring in file_selection_condition)]
#             else:
#                 filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring in path.stem for substring in file_selection_condition)]
#         else:
#             if negated_condition:
#                 filepaths = [path for path in Path(folder_path).glob('*.dat') if not file_selection_condition in path.stem]
#             else:
#                 filepaths = [path for path in Path(folder_path).glob('*.dat') if file_selection_condition in path.stem]
#         # Create list to hold data
#         list_of_df = []
#         # Create list to check if experiment was stopped during a measurement
#         list_of_n_measurements = []
#         # Define progress bar
#         pbar = tqdm(filepaths, desc='Loading data')
#         # Loop over all files
#         for file in pbar:
#             # Used to track which file is being read
#             file_name = file.name
#             # Used to extract the relevant chemical compound for the measurements
#             experiment_name = file.stem
#             # Update progress bar
#             pbar.set_postfix_str(f'Currently loading {file_name}')
#             # Detect size of header and column names
#             rows_to_skip = 0
#             column_names = []
#             # Open the .dat file
#             with open(file) as f:
#                 # Loop through the lines 1-by-1
#                 for line in f:
#                     # If the line does not start with a "#" and isn't blank we have reached the data and we end the loop.
#                     if '#' not in line and len(line) >= 10:
#                         break
#                     # If the line starts with a "#" or is blank we are in the header.
#                     elif '#' in line or len(line) < 10:
#                         # Count the number of rows to skip
#                         rows_to_skip += 1
#                         # Clean up the line
#                         line = line.replace('\n','').split(' ')
#                         # Extract the column names
#                         column_names = [column for column in line if column][1:]
#             # Read the .dat file into a dataframe
#             df = pd.read_csv(
#                 file, 
#                 sep=' ',
#                 header=None,
#                 names=column_names,
#                 skiprows=rows_to_skip,
#                 skip_blank_lines=True,
#                 on_bad_lines='skip',
#                 keep_default_na=False,
#                 )
#             # Convert the column values to floats
#             df[column_names] = df[column_names].apply(pd.to_numeric, errors='coerce', downcast='float')
#             # Remove any rows that contained non-numeric data
#             df.dropna(axis=0, inplace=True)
#             # Log the filename
#             df['Filename'] = file_name
#             # Do beamline/synchrotron specific things
#             if synchrotron in ['BALDER']:
#                 # The current measurement number.
#                 measurement = np.int(experiment_name.split('_')[-1])
#                 # Add the measurement and experiment information to the dataframe
#                 df['Measurement'] = measurement
#                 df['Experiment'] = '_'.join(experiment_name.split('_')[:-1])
#                 # Append dataframe to the list
#                 list_of_df.append(df)
#                 # Tags defining lines with timestamps
#                 start_tag = '#C Acquisition started'
#                 end_tag = '#C Acquisition ended'
#                 # Timestamp format
#                 time_format = '%a %b %d %H:%M:%S %Y'
#                 # Extract timestamps
#                 with open(file) as f:
#                     # Read all lines
#                     lines = f.readlines()
#                     # Find line with starting time
#                     start_time = [line[26:].replace('\n', '') for line in lines if start_tag in line][0]
#                     # Format the line into a datetime object
#                     start_time = datetime.strptime(start_time, time_format)
#                     # Save the start time
#                     df['Start Time'] = start_time
#                     # Find the line with ending time
#                     end_time = [line[24:].replace('\n', '') for line in lines if end_tag in line][0]
#                     # Format the line into a datetime object
#                     end_time = datetime.strptime(end_time, time_format)
#                     # Save the end time
#                     df['End Time'] = end_time
#             elif synchrotron in ['ESRF', 'BALDER_2']:
#                 # Detect what data belongs to different measurements
#                 # Find the difference in time since measurement started
#                 if synchrotron in ['ESRF']:
#                     difference = df['Htime'].diff()
#                 elif synchrotron in ['BALDER_2']:
#                     difference = df['dt'].diff()
#                     # Create lists to hold times
#                     list_start_times = []
#                     list_end_times = []
#                     # Tags defining lines with timestamps
#                     start_tag = '#C Acquisition started'
#                     end_tag = '#C Acquisition ended'
#                     # Timestamp format
#                     time_format = '%a %b %d %H:%M:%S %Y'
#                     # Extract timestamps
#                     with open(file) as f:
#                         # Read all lines
#                         lines = f.readlines()
#                         # Find line with starting time
#                         start_times = [line[26:].replace('\n', '') for line in lines if start_tag in line]
#                         # Format the line into a datetime object
#                         start_times = [datetime.strptime(time, time_format) for time in start_times]
#                         # Find the line with ending time
#                         end_times = [line[24:].replace('\n', '') for line in lines if end_tag in line]
#                         # Format the line into a datetime object
#                         end_times = [datetime.strptime(time, time_format) for time in end_times]
#                 # Create list to hold the measurement numbers
#                 measurement = []
#                 # The current measurement number. We start at 1
#                 measurement_number = 1
#                 # The first datapoint has no defined difference, so we add it to measurement 1 now
#                 measurement.append(measurement_number)
#                 # Loop over the differences in time since measurement started
#                 for diff_val in difference:
#                     # If the difference is negative we have started a new measurement
#                     if diff_val < 0:
#                         # Increase the current measurement number
#                         measurement_number += 1
#                         measurement.append(measurement_number)
#                     # If the difference is positive we are in the same measurement
#                     elif diff_val >= 0:
#                         measurement.append(measurement_number)
#                     if synchrotron in ['BALDER_2']:
#                         # Append the times
#                         list_start_times.append(start_times[measurement_number - 1])
#                         list_end_times.append(end_times[measurement_number - 1])
#                 # Add the measurement and experiment information to the dataframe
#                 df['Measurement'] = measurement
#                 if '_ref' in experiment_name:
#                     df['Experiment'] = '_'.join(experiment_name.split('_')[:-1])
#                 else:
#                     df['Experiment'] = experiment_name
#                 # Add times to the dataframe
#                 if synchrotron in ['BALDER_2']:
#                     df['Start Time'] = list_start_times
#                     df['End Time'] = list_end_times
#                 # Append dataframe to the list
#                 list_of_df.append(df)
#                 # Log the number of measurements
#                 list_of_n_measurements.append(measurement_number)
#         # Merge all the dataframes into one dataset
#         df = pd.concat(list_of_df)
#         # Reset the index
#         df.reset_index(drop=True, inplace=True)
#         # Log the number of measurements in each experiment
#         if synchrotron in ['BALDER']:
#             # Loop over the experiments
#             for experiment in df['Experiment'].unique():
#                 # Find the number of measurements in the experiment
#                 n_measurements = np.amax(df['Measurement'][df['Experiment'] == experiment])
#                 # Log the number of measurements in the experiment
#                 list_of_n_measurements.append(n_measurements)
#         # Remove incomplete measurements
#         if np.amin(list_of_n_measurements) != np.amax(list_of_n_measurements):
#             if verbose:
#                 print('Incomplete measurement detected!')
#                 print(f'Not all edges were measured {np.amax(list_of_n_measurements)} times, but only {np.amin(list_of_n_measurements)} times.')
#                 print('Incomplete measurements will be removed unless keep_incomplete="True".')
#             if not keep_incomplete:
#                 df.drop(df[df['Measurement'] > np.amin(list_of_n_measurements)].index, inplace=True)
#                 if verbose:
#                     print('\nIncomplete measurements were removed!')
#     return df
# #%% Saving data

# def save_data(
#     data: pd.DataFrame,
#     filename: str='XAS_data.csv',
#     save_folder: str='./Data/SavedData/',
# ) -> None:
#     """Function to save dataframe to .csv file for use in other analysis or plotting programs.

#     Args:
#         data (pd.DataFrame): Dataframe to save.
#         filename (optional, str): Filename to use for the saved file. Defaults to 'XAS_data.csv'.
#         save_folder (optional, str): Folder to save the file in. Defaults to './Data/SavedData/'.

#     Returns:
#         None
#     """    
#     # Create the save folder if it doesn't exist
#     Path(save_folder).mkdir(parents=True, exist_ok=True)
#     # Save the file
#     data.to_csv(save_folder + filename, index=False)
#     return None

# #%% Preprocessing functions

# def processing_df(
#     df: pd.DataFrame,
#     synchrotron: str,
#     metal: Union[str, None]=None,
#     precursor: Union[str, None]=None,
# ) -> pd.DataFrame:
#     """------------------------------------------------------------------
#     Initial preprocessing of XAS data.

#     Select relevant data columns, calculate absorption and transmission signals, and initialize empty columns used for normalization.

#     Args:
#         df (pd.DataFrame): The raw input data.
#         synchrotron (str): Name of the synchrotron the data was measured at.
#         metal (optional, Union[str, None]): The metal that was measured. Defaults to None.
#         precursor (optional, Union[str, None]): The precursor that was measured. Defaults to None.

#     Returns:
#         pd.DataFrame: The cleaned and preprocessed data.
#     """    
#      # List of the synchrotrons which the function can load data from
#     implemented = ['ESRF', 'BALDER', 'BALDER_2']
#     # Check the data format can be handled
#     assert synchrotron in implemented, f'Loading of data from {synchrotron} is not implemented. The implemented synchrotrons are:\n\t{implemented}\n\nIf you want the loading of data from a specific facility implemented contact me at ufj@chem.ku.dk or submit a request at [github link].'
#     if synchrotron in ['ESRF']:
#         # Select the relevant columns
#         df_new = df[['Filename', 'Experiment', 'Measurement', 'ZapEnergy', 'MonEx', 'xmap_roi00', 'Ion1']]
#         # Assign the measured metal or infer from file
#         if metal != None:
#             df_new['Metal'] = metal
#         else:
#             df_new['Metal'] = df['Experiment'].str[:2]
#         # Assign the measured precursor counter-ion or infer from file
#         if precursor != None:
#             df_new['Precursor'] = precursor
#         elif len(df['Experiment'][0]) > 2:
#             df_new['Precursor'] = df['Experiment'].str[2:]
#         else:
#             df_new['Precursor'] = precursor
#         # Calculate the correct x- and y-values for looking at the measured absorption and transmission data
#         df_new['Energy'] = df_new['ZapEnergy'] * 1000
#         df_new['Temperature'] = df['Nanodac']
#         df_new['Absorption'] = df_new['xmap_roi00'].to_numpy() / df_new['MonEx'].to_numpy()
#         df_new['Transmission'] = np.log(df_new['MonEx'].to_numpy() / df_new['Ion1'].to_numpy())
#     elif synchrotron in ['BALDER']:
#         # Select the relevant columns
#         df_new = df[['Filename', 'Experiment', 'Measurement', 'Start Time', 'End Time', 'albaem01_ch1', 'albaem01_ch2', 'albaem02_ch3', 'albaem02_ch4']]
#         # Assign the measured metal or infer from file
#         if metal != None:
#             df_new['Metal'] = metal
#             # Re-assign the experiment name
#             df_new['Experiment'] = metal
#         else:
#             raise ValueError('The measured metal can not be inferred from the data format at BALDER')
#         # Assign the measured precursor counter-ion. It can not be inferred from file
#         df_new['Precursor'] = precursor
#         if precursor != None:
#             # Re-assign the experiment name
#             df_new['Experiment'] += df_new['Precursor']
#         # Calculate the correct x- and y-values for looking at the measured absorption and transmission data
#         df_new['Relative Time'] = (df_new['Start Time'] - df_new['Start Time'][0]).dt.total_seconds()
#         df_new['Energy'] = df['mono1_energy']
#         df_new['Temperature'] = 0
#         df_new['Absorption'] = 0
#         df_new['Transmission'] = ( df['albaem01_ch1'] + df['albaem01_ch2'] ) / ( df['albaem02_ch3'] + df['albaem02_ch4'] )
#     elif synchrotron in ['BALDER_2']:
#         # Select the relevant columns
#         df_new = df[['Filename', 'Experiment', 'Measurement', 'Start Time', 'End Time', 'albaem02_ch1', 'albaem02_ch2', 'albaem02_ch3', 'albaem02_ch4']]
#         # Assign the measured metal or infer from file
#         if metal != None:
#             df_new['Metal'] = metal
#         else:
#             df_new['Metal'] = df['Experiment'].str[:2]
#         # Assign the measured precursor counter-ion. It can not be inferred from file
#         if precursor != None:
#             df_new['Precursor'] = precursor
#         elif len(df['Experiment'][0]) > 2:
#             df_new['Precursor'] = df['Experiment'].str[2:]
#         else:
#             df_new['Precursor'] = precursor
#         # Calculate the correct x- and y-values for looking at the measured absorption and transmission data
#         df_new['Relative Time'] = (df_new['Start Time'] - df_new['Start Time'][0]).dt.total_seconds()
#         df_new['Energy'] = df['mono1_energy']
#         df_new['Temperature'] = 0
#         df_new['Absorption'] = 0
#         df_new['Transmission'] = ( df['albaem02_ch1'] + df['albaem02_ch2'] ) / ( df['albaem02_ch3'] + df['albaem02_ch4'] )

#     df_new['Energy_Corrected'] = 0
#     df_new['Normalized'] = 0
#     df_new['pre_edge'] = 0
#     df_new['post_edge'] = 0
#     return df_new

# def calc_edge_correction(
#     df: pd.DataFrame, 
#     metal: str, 
#     edge: str, 
#     transmission: bool=False
# ) -> float:
#     """------------------------------------------------------------------
#     Calculation of the energy shift between the measured edge and the table value.

#     Args:
#         df (pd.DataFrame): The preprocessed data.
#         metal (str): The measured metal.
#         edge (str): The relevant edge (K, L1, L2, L3).
#         transmission (optional, bool): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.

#     Returns:
#         float: Edge energy shift
#     """    
#     # Choose what type of XAS data to work with
#     if transmission:
#         data_type = 'Transmission'
#     else:
#         data_type = 'Absorption'
#     # Find the table value for the relevant edge
#     edge_table = xray_edge(metal, edge, energy_only=True)
#     # Create dataframe filter
#     df_filter = (df['Experiment'].str.contains(metal)) & (df['Measurement'] == 1)
#     try:
#         # Find the measured edge energy
#         edge_measured = find_e0(df['Energy'][df_filter].to_numpy(), df[data_type][df_filter].to_numpy())
#         # Calculate the needed correction
#         correction_value = edge_table - edge_measured
#     except:
#         correction_value = None
#     return correction_value

# def normalize_data(
#     df: pd.DataFrame, 
#     edge_correction_energies: dict, 
#     subtract_preedge: bool=True, 
#     transmission: bool=False,
# ) -> None:
#     """------------------------------------------------------------------
#     Normalization of XAS data.

#     The subtraction of pre-edge fit from the data can ruin the normalization, so set subtract_preedge=False if the normalized data looks weird.

#     Args:
#         df (pd.DataFrame): The preprocessed data.
#         edge_correction_energies (dict): Energy shifts for all relevant edges.
#         subtract_preedge (optional, bool): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
#         transmission (optional, bool): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.

#     Returns:
#         None
#     """    
#     # Choose what type of XAS data to work with
#     if transmission:
#         data_type = 'Transmission'
#     else:
#         data_type = 'Absorption'
#     # Iterate over each experiment
#     for experiment in df['Experiment'].unique():
#         # Select only relevant values
#         exp_filter = (df['Experiment'] == experiment)
#         # Measured metal
#         metal = df['Metal'][exp_filter].to_list()[0]
#         # Correct for the energy shift at the edge
#         df['Energy_Corrected'][exp_filter] = df['Energy'][exp_filter] + edge_correction_energies[metal]
#         # Iterate over each measurement
#         for measurement_id in df['Measurement'][exp_filter].unique():
#             # Select only relevant values
#             df_filter = (df['Experiment'] == experiment) & (df['Measurement'] == measurement_id)
#             # Create dummy Group to collect post edge fit
#             g = Group(name='temp_group')
#             # Subtract minimum value from measurement
#             df['Normalized'][df_filter] = df[data_type][df_filter] - np.amin(df[data_type][df_filter])
#             try:
#                 # Fit to the pre- and post-edge
#                 pre_edge(df['Energy_Corrected'][df_filter].to_numpy(), df['Normalized'][df_filter].to_numpy(), group=g)
#                 # Save pre- and post-edge for plotting
#                 df['pre_edge'][df_filter] = g.pre_edge
#                 df['post_edge'][df_filter] = g.post_edge
#                 # Subtract pre-edge fit if specified
#                 if subtract_preedge:
#                     df['Normalized'][df_filter] -= g.pre_edge
#                     # Re-fit to the post-edge 
#                     pre_edge(df['Energy_Corrected'][df_filter].to_numpy(), df['Normalized'][df_filter].to_numpy(), group=g)
#                 # Normalize with the post edge fit
#                 df['Normalized'][df_filter] /= g.post_edge
#             except:
#                 print(f'Error occurred during normalization of data from measurement {measurement_id}. The error is most likely due to the measurement being stopped before completion.\nThe measurement (incl. all edges) has therefore been removed from the dataset and measurement numbers are corrected.')
#                 df.drop(df[(df['Measurement'] == measurement_id)].index, inplace=True)
#                 df['Measurement'][(df['Measurement'] > measurement_id)] = df['Measurement'][(df['Measurement'] > measurement_id)] - 1
#     return None

# def combine_datasets(
#     datasets: list[pd.DataFrame]
# ) -> pd.DataFrame:
#     """Combine two or more datasets from the same experiment

#     Args:
#         datasets (list[pd.DataFrame]): List of the datasets in the correct order

#     Returns:
#         pd.DataFrame: The combined dataset
#     """    
#     # Create empty list to hold dataframes
#     df_list = []
#     # Counter to update measurement values
#     measurement_counter = 0
#     # Loop over all the datasets
#     for df in datasets:
#         # Update the measurement number
#         df['Measurement'] += measurement_counter
#         # Update the measurement counter
#         measurement_counter = np.amax(df['Measurement'])
#         # Add dataframe to the list
#         df_list.append(df)
#     # Combine the dataframes
#     combined_df = pd.concat(df_list)
#     combined_df.reset_index(inplace=True)
#     return combined_df

# def average_measurements(
#     data: pd.DataFrame, 
#     measurements_to_average: Union[list[int], np.ndarray, range], 
# ) -> pd.DataFrame:
#     """Calculates the average XAS data for the specified measurements.

#     Args:
#         data (pd.DataFrame): Dataframe containing the XAS data.
#         measurements_to_average (Union[list[int], np.ndarray, range]): The measurements to be averaged.

#     Returns:
#         averaged_data (pd.DataFrame): Dataframe containing the averaged XAS data.
#     """
#     # Create list to hold dataframes for each experiment
#     list_of_df = []
#     # Loop over all experiments
#     for experiment in data['Experiment'].unique():
#         # Loop over all the frames to average
#         for measurement in measurements_to_average:
#             # Select only relevant values
#             data_filter = (data['Experiment'] == experiment) & (data['Measurement'] == measurement)
#             # Create array to store average
#             if measurement == measurements_to_average[0]:
#                 df_avg = data[data_filter].copy()
#                 avg_absorption = np.zeros_like(data['Absorption'][data_filter], dtype=np.float64)
#                 avg_transmission = np.zeros_like(data['Transmission'][data_filter], dtype=np.float64)
#             # Sum the measurements together
#             avg_absorption += data['Absorption'][data_filter].to_numpy()
#             avg_transmission += data['Transmission'][data_filter].to_numpy()
#         # Divide by the number of measurements used
#         n_measurements = float(len(measurements_to_average))
#         avg_absorption /= n_measurements
#         avg_transmission /= n_measurements
#         # Put the averaged data into the dataframe
#         df_avg['Absorption'] = avg_absorption
#         df_avg['Transmission'] = avg_transmission
#         # Select only relevant values
#         temp_filter = (data['Experiment'] == experiment) & (data['Measurement'].isin(measurements_to_average))
#         # Calculate the average temperature for the averaged data
#         df_avg['Temperature'] = data['Temperature'][temp_filter].mean()
#         # Set the measurement number to 1
#         df_avg['Measurement'] = 1
#         # Change precursor value to indicate these are averages
#         df_avg['Precursor'] = 'Avg'
#         # if np.amin(measurements_to_average) < 10:
#         #     df_avg['Experiment'] += df_avg['Precursor']
#         # Add the dataframe with the averaged data to list
#         list_of_df.append(df_avg)
#     # Combine all experiments into one dataframe
#     df_avg = pd.concat(list_of_df)
#     return df_avg

# #%% Fitting functions

# def linear_combination(
#     weights: Parameters, 
#     basis_functions: list[np.array],
# ) -> np.array:
#     """------------------------------------------------------------------
#     Calculates the linear combination of a set of basis functions.

#     Args:
#         weights (Parameters): The weights of the basis functions.
#         basis_functions (list[np.array]): The basis functions.

#     Returns:
#         np.array: The linear combination.
#     """    
#     # Unpack the weights from the Parameter instance
#     weights = list(weights.valuesdict().values()) 
#     # Create zero array for the linear combination
#     combination = np.zeros_like(basis_functions[0])
#     # Create variable to ensure sum of weights is 1
#     weight_sum = 0
#     # Loop to add the contributions to the linear combination together
#     # List of basis functions should be 1 element longer than the weights
#     for weight, basis_function in zip(weights, basis_functions):#[:-1]): 
#         # Addition of the individual contribution to the linear combination
#         combination += weight * basis_function
#         # Sum of the weights
#         weight_sum += weight
#     return combination

# def residual(
#     target: np.array, 
#     estimate: np.array
# ) -> np.array:
#     """------------------------------------------------------------------
#     Calculate residual between to arrays.

#     Args:
#         target (np.array): The target array.
#         estimate (np.array): The estimated array.

#     Returns:
#         np.array: The residual array.
#     """    
#     return target - estimate

# def fit_func(
#     weights: Parameters, 
#     basis_functions: list[np.array], 
#     data_to_fit: np.array
# ) -> np.array:
#     """------------------------------------------------------------------
#     Fitting function used by lmfit.minimize for fitting the weights in Linear Combination Analysis (LCA).

#     Args:
#         weights (Parameters): The weights of the basis functions.
#         basis_functions (list[np.array]): The basis functions.
#         data_to_fit (np.array): The measurement to be fitted by LCA.

#     Returns:
#         np.array: The residual between the measured data and the linear combination.
#     """    
#     # Calculate the linear combination of the basis functions
#     estimate = linear_combination(weights, basis_functions)
#     # Calculate the residual
#     return residual(data_to_fit, estimate)

# def linear_combination_analysis(
#     data: pd.DataFrame, 
#     products: pd.DataFrame, 
#     precursors: pd.DataFrame, 
#     intermediates: Union[pd.DataFrame, None]=None,
#     fit_min: Union[float, int, None]=0,
#     fit_max: Union[float, int, None]=np.infty,
#     verbose: bool=False, 
#     return_dataframe: bool=True
# ) -> Union[pd.DataFrame, list[MinimizerResult]]:
#     """------------------------------------------------------------------
#     Linear Combination Analysis (LCA) of an entire dataset. 

#     Provided with a dataset and standards of the relevant metal foils and precursors LCA is performed for all combinations of metals and their precursors on each measurement.

#     Args:
#         data (pd.DataFrame): The normalized data.
#         products (pd.DataFrame): The normalized product standards.
#         precursors (pd.DataFrame): The normalized precursor standards.
#         intermediate (optional, pd.DataFrame): The normalized intermediate standards.
#         fit_min (optional, Union[float, int, None]): The minimum energy value to include in the LCA fit. Defaults to 0.
#         fit_max (optional, Union[float, int, None]): The maximum energy value to include in the LCA fit. Defaults to infinity.
#         verbose (optional, bool): Boolean flag controlling if the fit results are printed. Defaults to False.
#         return_dataframe (optional, bool): Boolean flag controlling if a dataframe (True) or a list of fit result objects (False) are returned. Defaults to True.

#     Returns:
#         pd.DataFrame | list[MinimizerResult]: The results of the LCA on the dataset as either a dataframe or a list of MinimizerResult objects.
#     """    
#     # List to hold lmfit output objects
#     fit_results = []
#     # Lists to hold values for Dataframe
#     list_experiments = []
#     list_metals = []
#     list_products = []
#     list_intermediates = []
#     list_precursors = []
#     list_precursor_types = []
#     list_measurements = []
#     list_temperatures = []
#     list_parameters = []
#     list_values = []
#     list_stderrors = []
#     list_stderrors_corrected = []
#     list_energy_range = []
#     list_basis_functions = []
#     # Calculate the number of combinations to perform LCA for
#     metals = data['Metal'].unique()
#     n_combinations = 0
#     for metal in metals:
#         if intermediates == None:
#             n_combinations += len(precursors['Experiment'][precursors['Metal'] == metal].unique()) * len(products['Experiment'][products['Metal'] == metal].unique())
#         else:
#             n_combinations += len(precursors['Experiment'][precursors['Metal'] == metal].unique()) * len(products['Experiment'][products['Metal'] == metal].unique()) * (len(intermediates['Experiment'][products['Metal'] == metal].unique()) + 1)
#     # Progress bar for the LCA progress
#     with tqdm(data['Metal'].unique(), total=n_combinations, desc='LCA progress: ') as pbar_metal:
#         # Loop over all metal edges
#         for metal in data['Metal'].unique():
#             # Loop over all relevant precursors
#             relevant_precursors = precursors['Experiment'][precursors['Metal'] == metal].unique()
#             for precursor in relevant_precursors:
#                 # Loop over all relevant products
#                 relevant_products = products['Experiment'][products['Metal'] == metal].unique()
#                 for product in relevant_products:
#                     # Initialize intermediate as None
#                     intermediate = None
#                     # Update descriptive text on progress bar
#                     pbar_metal.set_postfix_str(f'Analysing {precursor} + {product}')
#                     # Loop over all measurements
#                     with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'{product} + {precursor}', mininterval=0.01) as pbar_measurement:
#                         for measurement in data['Measurement'][data['Metal'] == metal].unique():
#                             # Check if the metal exists in the dataset
#                             # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
#                             # Create filter for relevant values
#                             data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
#                             # Extract the energy range covered by a measurement
#                             data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
#                             # Extract the relevant data
#                             data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)]
#                             # Extract the temperature
#                             temperature = np.median(data['Temperature'][data_filter])
#                             # Check if the metal foil exists in the dataset
#                             assert product in  products['Experiment'].unique(), f'No metal foil standard with the name: {product}\n\nValid values are: {products.Experiment.unique()}'
#                             # Create filter for relevant values
#                             product_filter = (products['Metal'] == metal) & (products['Measurement'] == 1)
#                             # Extract the relevant data
#                             product_basis = np.interp(data_range, products['Energy_Corrected'][product_filter], products['Normalized'][product_filter])
#                             # Check if the precursor exists in the dataset
#                             assert precursor in precursors['Experiment'].unique(), f'No precursor standard with the name: {precursor}\n\nValid values are: {precursors.Experiment.unique()}'
#                             # Create filter for relevant values
#                             precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
#                             # Extract the relevant data
#                             precursor_basis = np.interp(data_range, precursors['Energy_Corrected'][precursor_filter], precursors['Normalized'][precursor_filter])
#                             # Group the two basis functions
#                             basis_functions = [product_basis, precursor_basis]
#                             # Initialize the fit parameters
#                             fit_params = Parameters()
#                             fit_params.add('product_weight', value=0.5, min=0, max=1)
#                             fit_params.add('precursor_weight', expr='1.0 - product_weight', min=0, max=1)
#                             try:
#                                 # Fit the linear combination of basis functions to the measurement
#                                 fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
#                                 # Store the fit results as lmfit MinimizerResults
#                                 fit_results.append(fit_out)
#                             except:
#                                 raise Exception(f'Error occurred when fitting measurement {measurement} of {precursor} + {product}.')
#                             # Save the values needed for the results Dataframe
#                             for name, param in fit_out.params.items():
#                                 list_experiments.append(f'{precursor} + {product}')
#                                 list_metals.append(metal)
#                                 list_products.append(product)
#                                 list_intermediates.append(intermediate)
#                                 list_precursors.append(precursor)
#                                 list_precursor_types.append(precursor[2:])
#                                 list_measurements.append(measurement)
#                                 list_temperatures.append(temperature)
#                                 list_parameters.append(name)
#                                 list_values.append(param.value)
#                                 list_stderrors.append(param.stderr)
#                                 list_energy_range.append(data_range)
#                                 # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
#                                 if name == 'precursor_weight' and param.value >= 0.999:
#                                     list_stderrors_corrected.append(list_stderrors[-2])
#                                 else:
#                                     list_stderrors_corrected.append(param.stderr)
#                             list_basis_functions.extend(basis_functions)
#                             # Check if the fit results should be printed
#                             if verbose:
#                                 print(f'Fit results for the linear combination of {product} and {precursor} to measurement {measurement} of the {metal} edge.\n')
#                                 print(fit_report(fit_out))
#                                 print('\n')
#                     # Update pbar
#                     pbar_metal.update()
#                     if intermediates != None:
#                         # Loop over all relevant intermediates
#                         relevant_intermediates = intermediates['Experiment'][intermediates['Metal'] == metal].unique()
#                         for intermediate in relevant_intermediates:
#                             # Update descriptive text on progress bar
#                             pbar_metal.set_postfix_str(f'Analysing {precursor} + {product} + {intermediate}')
#                             # Loop over all measurements
#                             with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'{precursor} + {product} + {intermediate}', mininterval=0.01) as pbar_measurement:
#                                 for measurement in pbar_measurement:
#                                     # Check if the metal exists in the dataset
#                                     # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
#                                     # Create filter for relevant values
#                                     data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
#                                     # Extract the energy range covered by a measurement
#                                     data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
#                                     # Extract the relevant data
#                                     data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)]
#                                     # Extract the temperature
#                                     temperature = np.median(data['Temperature'][data_filter])
#                                     # Check if the metal foil exists in the dataset
#                                     assert product in products['Experiment'].unique(), f'No product standard with the name: {product}\n\nValid values are: {products.Experiment.unique()}'
#                                     # Create filter for relevant values
#                                     product_filter = (products['Experiment'].str.contains(metal)) & (products['Measurement'] == 1)
#                                     # Extract the relevant data
#                                     product_basis = np.interp(data_range, products['Energy_Corrected'][product_filter], products['Normalized'][product_filter])
#                                     # Check if the intermediate exists in the dataset
#                                     assert intermediate in intermediates['Experiment'].unique(), f'No precursor standard with the name: {intermediate}\n\nValid values are: {intermediates.Experiment.unique()}'
#                                     # Create filter for relevant values
#                                     intermediate_filter = (intermediates['Experiment'] == intermediate) & (intermediates['Measurement'] == 1)
#                                     # Extract the relevant data
#                                     intermediate_basis = np.interp(data_range, intermediates['Energy_Corrected'][intermediate_filter], intermediates['Normalized'][intermediate_filter])
#                                     # Check if the precursor exists in the dataset
#                                     assert precursor in precursors['Experiment'].unique(), f'No precursor standard with the name: {precursor}\n\nValid values are: {precursors.Experiment.unique()}'
#                                     # Create filter for relevant values
#                                     precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
#                                     # Extract the relevant data
#                                     precursor_basis = np.interp(data_range, precursors['Energy_Corrected'][precursor_filter], precursors['Normalized'][precursor_filter])
#                                     # Group the two basis functions
#                                     basis_functions = [product_basis, intermediate_basis, precursor_basis]
#                                     # Initialize the fit parameters
#                                     fit_params = Parameters()
#                                     fit_params.add('product_weight', value=0.33, min=0, max=1)
#                                     fit_params.add('intermediate_weight', value=0.33, min=0, max=1)
#                                     fit_params.add('precursor_weight', expr='1.0 - product_weight - intermediate_weight', min=0, max=1)
#                                     try:
#                                         # Fit the linear combination of basis functions to the measurement
#                                         fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
#                                         # Store the fit results as lmfit MinimizerResults
#                                         fit_results.append(fit_out)
#                                     except:
#                                         raise Exception(f'Error occurred when fitting measurement {measurement} of {precursor} + {product} + {intermediate}.')
#                                     # Save the values needed for the results Dataframe
#                                     for name, param in fit_out.params.items():
#                                         list_experiments.append(f'{precursor} + {intermediate} + {product}')
#                                         list_metals.append(metal)
#                                         list_products.append(product)
#                                         list_intermediates.append(intermediate)
#                                         list_precursors.append(precursor)
#                                         list_precursor_types.append(precursor[2:])
#                                         list_measurements.append(measurement)
#                                         list_temperatures.append(temperature)
#                                         list_parameters.append(name)
#                                         list_values.append(param.value)
#                                         list_stderrors.append(param.stderr)
#                                         list_energy_range.append(data_range)
#                                         # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to themaximum allowed values
#                                         if name == 'precursor_weight' and param.value >= 0.999:
#                                             list_stderrors_corrected.append(list_stderrors[-2])
#                                         else:
#                                             list_stderrors_corrected.append(param.stderr)
#                                     list_basis_functions.extend(basis_functions)
#                                     # Check if the fit results should be printed
#                                     if verbose:
#                                         print(f'Fit results for the linear combination of {product}, {intermediate} and {precursor} to measurement {measurement} of the {metal} edge.\n')
#                                         print(fit_report(fit_out))
#                                         print('\n')
#                             # Update pbar
#                             pbar_metal.update()
#     # Save the fit results as a Dataframe
#     df_results = pd.DataFrame({
#         'Experiment':list_experiments,
#         'Metal':list_metals,
#         'Product':list_products,
#         'Intermediate':list_intermediates,
#         'Precursor':list_precursors,
#         'Precursor Type': list_precursor_types,
#         'Measurement':list_measurements,
#         'Temperature':list_temperatures,
#         'Temperature Average': 0,
#         'Temperature Std': 0,
#         'Parameter':list_parameters,
#         'Value':list_values,
#         'StdErr':list_stderrors,
#         'StdCorrected':list_stderrors_corrected,
#         'Energy Range': list_energy_range,
#         'Basis Function':list_basis_functions,
#         })
#     # Calculate average and standard deviation of the temperature across all edges within 1 measurement
#     for measurement in df_results['Measurement'].unique():
#         # Create filter for relvant values
#         df_filter = (df_results['Parameter'] == 'product_weight') & (df_results['Precursor Type'] == df_results['Precursor Type'].mode().to_list()[0])
#         meas_filter = (df_results['Measurement'] == measurement)
#         # Calculate the mean
#         avg_temp = np.mean(df_results['Temperature'][df_filter & meas_filter])
#         df_results['Temperature Average'][meas_filter] = avg_temp 
#         # Calculate the standard deviation
#         std_temp = np.std(df_results['Temperature'][df_filter & meas_filter], ddof=1)
#         df_results['Temperature Std'][meas_filter] = std_temp 
#     # Replace NaN values with interpolated values
#     df_results = df_results.interpolate()
#     # Check if the dataframe or lmfit MinimizerResults should be returned
#     if return_dataframe:
#         return df_results
#     else:
#         return fit_results

# def LCA_internal(
#     data: pd.DataFrame, 
#     initial_state_index: int=0, 
#     final_state_index: int=-1, 
#     intermediate_state_index: Union[int, None]=None,
#     fit_min: Union[float, int, None]=None,
#     fit_max: Union[float, int, None]=None,
#     verbose: bool=False, 
#     return_dataframe: bool=True
# ) -> Union[pd.DataFrame, list[MinimizerResult]]:
#     """------------------------------------------------------------------
#     Linear Combination Analysis (LCA) of an experiment using measurements as the components. 

#     Provided with a dataset and standards of the relevant metal foils and precursors LCA is performed for all combinations of metals and their precursors on each measurement.

#     Args:
#         data (pd.DataFrame): The normalized data.
#         initial_state_index (int): The measurement to use as a reference for the initial state.
#         final_state_index (int): The measurement to use as a reference for the final state.
#         intermediate_state_index (optional, Union[int, None]): The measurement to use as a reference for the intermediate state. Defaults to None.
#         fit_min (optional, Union[float, int, None]): The minimum energy value to include in the LCA fit. Defaults to 0.
#         fit_max (optional, Union[float, int, None]): The maximum energy value to include in the LCA fit. Defaults to infinity.
#         verbose (optional, bool): Boolean flag controlling if the fit results are printed. Defaults to False.
#         return_dataframe (optional, bool): Boolean flag controlling if a dataframe (True) or a list of fit result objects (False) are returned. Defaults to True.

#     Returns:
#         pd.DataFrame | list[MinimizerResult]: The results of the LCA on the dataset as either a dataframe or a list of MinimizerResult objects.
#     """    
#     # List to hold lmfit output objects
#     fit_results = []
#     # Lists to hold values for Dataframe
#     list_experiments = []
#     list_metals = []
#     list_products = []
#     list_intermediates = []
#     list_precursors = []
#     list_precursor_types = []
#     list_measurements = []
#     list_temperatures = []
#     list_parameters = []
#     list_values = []
#     list_stderrors = []
#     list_stderrors_corrected = []
#     list_energy_range = []
#     list_basis_functions = []
#     # Create progress bar for LCA progress
#     with tqdm(data['Metal'].unique(), desc='LCA progress: ') as pbar_metal:
#         # Loop over all metal edges
#         for metal in data['Metal'].unique():
#             if intermediate_state_index == None:
#                 # Update descriptive text on progress bar
#                 pbar_metal.set_postfix_str(f'Analysing frame {initial_state_index} + {final_state_index}')
#                 # Select relevant values
#                 reference_filter = (data['Metal'] == metal) & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)
#                 # Select the inital state measurement
#                 initial_basis = data['Normalized'][reference_filter & (data['Measurement'] == initial_state_index)].to_numpy()
#                 # Select the final state measurement
#                 final_basis = data['Normalized'][reference_filter & (data['Measurement'] == final_state_index)].to_numpy()
#                 # Loop over all measurements
#                 with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'Frame {initial_state_index} + {final_state_index}', mininterval=0.01) as pbar_measurement:
#                     for measurement in data['Measurement'][data['Metal'] == metal].unique():
#                         # Check if the metal exists in the dataset
#                         # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
#                         # Create filter for relevant values
#                         data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
#                         # Extract the energy range covered by a measurement
#                         data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
#                         # Extract the relevant data
#                         data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)].to_numpy()
#                         # Extract the temperature
#                         temperature = np.median(data['Temperature'][data_filter])
#                         # Group the two basis functions
#                         basis_functions = [final_basis, initial_basis]
#                         # Initialize the fit parameters
#                         fit_params = Parameters()
#                         fit_params.add('product_weight', value=0.5, min=0, max=1)
#                         fit_params.add('precursor_weight', expr='1.0 - product_weight', min=0, max=1)
#                         try:
#                             # Fit the linear combination of basis functions to the measurement
#                             fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
#                             # Store the fit results as lmfit MinimizerResults
#                             fit_results.append(fit_out)
#                         except:
#                             raise Exception(f'Error occurred when fitting measurement {measurement} with frame {initial_state_index} + {final_state_index}.')
#                         # Save the values needed for the results Dataframe
#                         for name, param in fit_out.params.items():
#                             list_experiments.append(f'Frame {initial_state_index} + {final_state_index}')
#                             list_metals.append(metal)
#                             list_products.append(final_state_index)
#                             list_intermediates.append(intermediate_state_index)
#                             list_precursors.append(initial_state_index)
#                             list_precursor_types.append('Internal')
#                             list_measurements.append(measurement)
#                             list_temperatures.append(temperature)
#                             list_parameters.append(name)
#                             list_values.append(param.value)
#                             list_stderrors.append(param.stderr)
#                             list_energy_range.append(data_range)
#                             # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
#                             if name == 'precursor_weight' and (param.value >= 0.99 or param.stderr > 1.):
#                                 list_stderrors_corrected.append(list_stderrors[-2])
#                             else:
#                                 list_stderrors_corrected.append(param.stderr)
#                         list_basis_functions.extend(basis_functions)
#                         # Check if the fit results should be printed
#                         if verbose:
#                             print(f'Fit results for the linear combination of frame {initial_state_index} and {final_state_index} to measurement {measurement} of the {metal} edge.\n')
#                             print(fit_report(fit_out))
#                             print('\n')
#                         # Update pbar
#                         pbar_metal.update()
#             elif intermediate_state_index != None:
#                 # Update descriptive text on progress bar
#                 pbar_metal.set_postfix_str(f'Analysing frame {initial_state_index} + {intermediate_state_index} + {final_state_index}')
#                 # Select relevant values
#                 reference_filter = (data['Metal'] == metal) & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)
#                 # Select the inital state measurement
#                 initial_basis = data['Normalized'][reference_filter & (data['Measurement'] == initial_state_index)].to_numpy()
#                 # Select the intermediate state measurement
#                 intermediate_basis = data['Normalized'][reference_filter & (data['Measurement'] == intermediate_state_index)].to_numpy()
#                 # Select the final state measurement
#                 final_basis = data['Normalized'][reference_filter & (data['Measurement'] == final_state_index)].to_numpy()
#                 # Loop over all measurements
#                 with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'Frame {initial_state_index} + {intermediate_state_index} + {final_state_index}', mininterval=0.01) as pbar_measurement:
#                     for measurement in pbar_measurement:
#                         # Check if the metal exists in the dataset
#                         # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
#                         # Create filter for relevant values
#                         data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
#                         # Extract the energy range covered by a measurement
#                         data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
#                         # Extract the relevant data
#                         data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)].to_numpy()
#                         # Extract the temperature
#                         temperature = np.median(data['Temperature'][data_filter])
#                         # Group the basis functions
#                         basis_functions = [final_basis, intermediate_basis, initial_basis]
#                         # Initialize the fit parameters
#                         fit_params = Parameters()
#                         fit_params.add('product_weight', value=0.33, min=0, max=1)
#                         fit_params.add('intermediate_weight', value=0.33, min=0, max=1)
#                         fit_params.add('precursor_weight', expr='1.0 - product_weight - intermediate_weight', min=0, max=1)
#                         try:
#                             # Fit the linear combination of basis functions to the measurement
#                             fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
#                             # Store the fit results as lmfit MinimizerResults
#                             fit_results.append(fit_out)
#                         except:
#                             raise Exception(f'Error occurred when fitting measurement {measurement} with frame {initial_state_index} + {intermediate_state_index} + {final_state_index}.')
#                         # Save the values needed for the results Dataframe
#                         for name, param in fit_out.params.items():
#                             list_experiments.append(f'Frame {initial_state_index} + {intermediate_state_index} + {final_state_index}')
#                             list_metals.append(metal)
#                             list_products.append(final_state_index)
#                             list_intermediates.append(intermediate_state_index)
#                             list_precursors.append(initial_state_index)
#                             list_precursor_types.append('Internal')
#                             list_measurements.append(measurement)
#                             list_temperatures.append(temperature)
#                             list_parameters.append(name)
#                             list_values.append(param.value)
#                             list_stderrors.append(param.stderr)
#                             list_energy_range.append(data_range)
#                             # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
#                             if name == 'precursor_weight' and (param.value >= 0.99 or param.stderr > 0.5):
#                                 list_stderrors_corrected.append(np.mean(list_stderrors_corrected[-2:]))
#                             elif name == 'intermediate_weight' and (param.value <= 0.01 and param.stderr > 1.):
#                                 list_stderrors_corrected.append(np.mean(list_stderrors_corrected[-1]))
#                             else:
#                                 list_stderrors_corrected.append(param.stderr)
#                         list_basis_functions.extend(basis_functions)
#                         # Check if the fit results should be printed
#                         if verbose:
#                             print(f'Fit results for the linear combination of frame {initial_state_index}, {intermediate_state_index} and {final_state_index} to measurement {measurement} of the {metal} edge.\n')
#                             print(fit_report(fit_out))
#                             print('\n')
#                 # Update pbar
#                 pbar_metal.update()
#     # Save the fit results as a Dataframe
#     df_results = pd.DataFrame({
#         'Experiment':list_experiments,
#         'Metal':list_metals,
#         'Product':list_products,
#         'Intermediate':list_intermediates,
#         'Precursor':list_precursors,
#         'Precursor Type': list_precursor_types,
#         'Measurement':list_measurements,
#         'Temperature':list_temperatures,
#         'Temperature Average': 0,
#         'Temperature Std': 0,
#         'Parameter':list_parameters,
#         'Value':list_values,
#         'StdErr':list_stderrors,
#         'StdCorrected':list_stderrors_corrected,
#         'Energy Range': list_energy_range,
#         'Basis Function':list_basis_functions,
#         })
#     # Calculate average and standard deviation of the temperature across all edges within 1 measurement
#     for measurement in df_results['Measurement'].unique():
#         # Create filter for relvant values
#         df_filter = (df_results['Parameter'] == 'product_weight') & (df_results['Precursor Type'] == df_results['Precursor Type'].mode().to_list()[0])
#         meas_filter = (df_results['Measurement'] == measurement)
#         # Calculate the mean
#         avg_temp = np.mean(df_results['Temperature'][df_filter & meas_filter])
#         df_results['Temperature Average'][meas_filter] = avg_temp 
#         # Calculate the standard deviation
#         std_temp = np.std(df_results['Temperature'][df_filter & meas_filter], ddof=1)
#         df_results['Temperature Std'][meas_filter] = std_temp 
#     # Replace NaN values with interpolated values
#     df_results = df_results.interpolate()
#     # Check if the dataframe or lmfit MinimizerResults should be returned
#     if return_dataframe:
#         return df_results
#     else:
#         return fit_results

# #%% Plotting functions

# def plot_non_normalized_xas(
#     df: pd.DataFrame, 
#     experiment: str, 
#     transmission: bool=False, 
#     pre_edge: bool=False, 
#     post_edge: bool=False, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='raw_xas_plot.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting of non-normalized XAS data.

#     The energy shift from the measurements is corrected in the plotted data.

#     Args:
#         df (pd.DataFrame): The dataset after normalization.
#         experiment (str): The metal, metal foil or precursor that was measured.
#         transmission (optional, bool): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
#         pre_edge (optional, bool): Boolean flag controlling if the pre-edge is plotted. Defaults to False.
#         post_edge (optional, bool): Boolean flag controlling if the post-edge is plotted. Defaults to False.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'raw_xas_plot.png'.

#     Returns:
#         None
#     """    
#     # Check if the experiment exists in the dataset
#     assert experiment in df.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {df.Experiment.unique()}'
#     # Choose what type of XAS data to work with
#     if transmission:
#         data_type = 'Transmission'
#     else:
#         data_type = 'Absorption'
#     # Create filter for relevant values
#     df_filter = (df['Experiment'] == experiment) & (df['Measurement'] == 1)
#     # Extract the minimum value in the data
#     min_value = np.amin(df[data_type][df_filter])
#     # Extract the number of measurements
#     n_measurements = int(np.amax(df['Measurement'][(df['Experiment'] == experiment)]))
#     if n_measurements == 1:
#         n_measurements += 1
#     # Determine what to use as x-axis
#     if df['Energy_Corrected'][df_filter].sum() > 0.:
#         x_column = 'Energy_Corrected'
#     else: 
#         x_column = 'Energy'
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Plot the measurements of the selected experiment/edge
#         sns.lineplot(
#             data=df[(df['Experiment'] == experiment)], 
#             x=x_column, 
#             y=data_type, 
#             hue='Measurement', 
#             palette='viridis',
#             )
#         # Plot the pre-edge fit 
#         if pre_edge:
#             sns.lineplot(
#                 x=df[x_column][df_filter], 
#                 y=df['pre_edge'][df_filter] + min_value,  
#                 color='r',
#                 linewidth=3,
#                 label='Pre-edge'
#                 )
#         # Plot the post-edge
#         if post_edge:
#             sns.lineplot(
#                 x=df[x_column][df_filter], 
#                 y=df['post_edge'][df_filter] + min_value,  
#                 color='k',
#                 linewidth=3,
#                 label='Post-edge'
#                 )
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Energy [eV]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             r'X-ray transmission, µ(E)$\cdot$x', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend( 
#             title='Measurement', 
#             fontsize=12, 
#             title_fontsize=13, 
#             ncol=1
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive figure using plotly
#     elif interactive:
#         # Formatting of the hover label "title"
#         x_formatting = '.0f'
#         # Plot the measurements of the selected experiment/edge
#         fig = px.line(
#             data_frame=df[(df['Experiment'] == experiment)],
#             x=x_column,
#             y=data_type,
#             color='Measurement',
#             color_discrete_sequence=px.colors.sample_colorscale('viridis', samplepoints=n_measurements),
#         )
#         # Change line formatting
#         fig.update_traces(
#             line=dict(
#                 width=2,
#             ),
#             xhoverformat=x_formatting,
#         )
#         # Plot the pre-edge fit 
#         if pre_edge:
#             fig.add_trace(
#                 go.Scatter(
#                     x=df[x_column][df_filter], 
#                     y=df['pre_edge'][df_filter] + min_value, 
#                     mode='lines',
#                     name='Pre-edge',
#                     line=dict(
#                         width=3,
#                         color='red',
#                     ),
#                     xhoverformat=x_formatting,
#                 ))
#         # Plot the post-edge
#         if post_edge:
#             fig.add_trace(
#                 go.Scatter(
#                     x=df[x_column][df_filter], 
#                     y=df['post_edge'][df_filter] + min_value, 
#                     mode='lines',
#                     name='Post-edge',
#                     line=dict(
#                         width=3,
#                         color='black',
#                     ),
#                     xhoverformat=x_formatting,
#                 ))
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Energy [eV]</b>',
#             yaxis_title='<b>'+data_type+'</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#         )
#         # Customize the hover labels
#         hovertemplate = '<br>Absorption = %{y:.2f} <br>Energy = %{x:.0f} eV'
#         fig.update_traces(hovertemplate=hovertemplate)
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_data(
#     data: pd.DataFrame, 
#     metal: str, 
#     foils: Union[pd.DataFrame, None]=None, 
#     products: Union[pd.DataFrame, None]=None, 
#     intermediates: Union[pd.DataFrame, None]=None, 
#     precursors: Union[pd.DataFrame, None]=None, 
#     precursor_suffix: Union[str, None]=None, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='xas_data.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting of normalized XAS data.

#     The measured standards can be shown in the same plot if the datasets with metal foils and precursors are provided.

#     Args:
#         data (pd.DataFrame): The dataset.
#         metal (str): The metal that should be plotted.
#         foils (optional, Union[pd.DataFrame | None]): Dataset of metal foils. Defaults to None.
#         products (optional, Union[pd.DataFrame | None]): Dataset of products. Defaults to None.
#         intermediates (optional, Union[pd.DataFrame | None]): Dataset of intermediates. Defaults to None.
#         precursors (optional, Union[pd.DataFrame, None]): Dataset of precursors. Defaults to None.
#         precursor_suffix (optional, Union[str, None]): The counter-ion used in a specific precursor that should be plotted. Defaults to None.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'xas_data.png'.

#     Returns:
#         None
#     """    
#     # Check if metal exists in the dataset
#     assert metal in data.Metal.unique(), f'No metal with the name: {metal}\n\nValid values are: {data.Metal.unique()}'
#     # Extract the number of measurements
#     n_measurements = int(np.amax(data['Measurement']))
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Plot metal foil if provided
#         if type(foils) != type(None):
#             # Check if the metal foil exists in the dataset
#             assert metal in foils.Metal.unique(), f'No precursor containing: {metal}\n\nValid values are: {foils.Experiment.unique()}'
#             sns.lineplot(
#                 data=foils[(foils['Metal'] == metal) & (foils['Measurement'] == 1)], 
#                 x='Energy_Corrected', 
#                 y='Normalized', 
#                 color='k', 
#                 linewidth=3, 
#                 label=metal+' foil'
#                 )
#         # Plot precursor(s) if provided
#         if type(precursors) != type(None):
#             # Plot only specific precursor if it is provided
#             if type(precursor_suffix) != type(None):
#                 # Check if the specified precursor exists in the dataset
#                 assert metal+precursor_suffix in precursors.Experiment.unique(), f'No metal with the name: {metal+precursor_suffix}\n\nValid values are: {precursors.Experiment.unique()}'
#                 sns.lineplot(
#                     data=precursors[(precursors['Experiment'] == metal+precursor_suffix) & (precursors['Measurement'] == 1)], 
#                     x='Energy_Corrected', 
#                     y='Normalized', 
#                     color='r', 
#                     linewidth=3, 
#                     label=metal+precursor_suffix
#                     )
#             # Plot all precursors containing the specified metal if no specific precursor is provided 
#             else:
#                 sns.lineplot(
#                     data=precursors[(precursors['Metal'] == metal) & (precursors['Measurement'] == 1)], 
#                     x='Energy_Corrected', 
#                     y='Normalized', 
#                     hue='Experiment', 
#                     linewidth=3,
#                     palette='colorblind'
#                     )
#             if type(intermediates) != type(None):
#                 sns.lineplot(
#                     data=intermediates[(intermediates['Metal'] == metal) & (intermediates['Measurement'] == 1)], 
#                     x='Energy_Corrected', 
#                     y='Normalized', 
#                     hue='Experiment', 
#                     linewidth=3,
#                     palette='colorblind'
#                     )
#             if type(products) != type(None):
#                 sns.lineplot(
#                     data=products[(products['Metal'] == metal) & (products['Measurement'] == 1)], 
#                     x='Energy_Corrected', 
#                     y='Normalized', 
#                     hue='Experiment', 
#                     linewidth=3,
#                     palette='colorblind'
#                     )
#         # Plot all measurements of specified metal edge
#         sns.lineplot(
#             data=data[(data['Metal'] == metal)], 
#             x='Energy_Corrected', 
#             y='Normalized', 
#             hue='Measurement', 
#             palette='viridis',
#             )
#         # Set limits of x-axis to match the edge measurements
#         plt.xlim(
#             (np.amin(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == 1)]), 
#             np.amax(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == 1)]))
#             )
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Energy [eV]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             'Normalized', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend(
#             loc='lower right', 
#             title='Measurement', 
#             fontsize=12, 
#             title_fontsize=13, 
#             ncol=1
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Formatting of the hover label "title"
#         x_formatting = '.0f'
#         # Plot the measurements of the selected experiment/edge
#         fig = px.line(
#             data_frame=data[(data['Metal'] == metal)],
#             x='Energy_Corrected',
#             y='Normalized',
#             color='Measurement',
#             color_discrete_sequence=px.colors.sample_colorscale('viridis', samplepoints=n_measurements),
#         )
#         # Change line formatting
#         fig.update_traces(
#             line=dict(
#                 width=2,
#             ),
#             xhoverformat=x_formatting,
#         )
#         # Plot metal foil if provided
#         if type(foils) != type(None):
#             # Check if the metal foil exists in the dataset
#             assert metal in foils.Metal.unique(), f'No foil with the metal: {metal}\n\nValid values are: {foils.Metal.unique()}'
#             foil_filter = (foils['Metal']== metal) & (foils['Measurement'] == 1)
#             fig.add_trace(
#                 go.Scatter(
#                     x=foils['Energy_Corrected'][foil_filter], 
#                     y=foils['Normalized'][foil_filter], 
#                     mode='lines',
#                     name=metal+' foil',
#                     line=dict(
#                         width=3,
#                         color='black',
#                     ),
#                     xhoverformat=x_formatting,
#                 ))
#         # Plot precursor(s) if provided
#         if type(precursors) != type(None):
#             # Plot only specific precursor if it is provided
#             if type(precursor_suffix) != type(None):
#                 # Check if the specified precursor exists in the dataset
#                 assert metal+precursor_suffix in precursors.Experiment.unique(), f'No metal with the name: {metal+precursor_suffix}\n\nValid values are: {precursors.Experiment.unique()}'
#                 precursor_filter = (precursors['Experiment'] == metal+precursor_suffix) & (precursors['Measurement'] == 1)
#                 fig.add_trace(
#                     go.Scatter(
#                         x=precursors['Energy_Corrected'][precursor_filter], 
#                         y=precursors['Normalized'][precursor_filter], 
#                         mode='lines',
#                         name=metal+precursor_suffix,
#                         line=dict(
#                             width=3,
#                             color='red',
#                         ),
#                         xhoverformat=x_formatting,
#                     ))
#             # Plot all precursors containing the specified metal if no specific precursor is provided 
#             else:
#                 for i, precursor in enumerate(precursors['Experiment'][precursors['Metal'] == metal].unique()):
#                     precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
#                     fig.add_trace(
#                         go.Scatter(
#                             x=precursors['Energy_Corrected'][precursor_filter], 
#                             y=precursors['Normalized'][precursor_filter], 
#                             mode='lines',
#                             name=precursor,
#                             line=dict(
#                                 width=3,
#                                 color=px.colors.qualitative.D3[i],
#                             ),
#                             xhoverformat=x_formatting,
#                         ))
#         if type(products) != type(None):
#             for i, product in enumerate(products['Experiment'][products['Metal'] == metal].unique()):
#                 product_filter = (products['Experiment'] == product) & (products['Measurement'] == 1)
#                 fig.add_trace(
#                     go.Scatter(
#                         x=products['Energy_Corrected'][product_filter], 
#                         y=products['Normalized'][product_filter], 
#                         mode='lines',
#                         name=product,
#                         line=dict(
#                             width=3,
#                             color=px.colors.qualitative.D3[i],
#                         ),
#                         xhoverformat=x_formatting,
#                     ))
#         if type(intermediates) != type(None):
#             for i, intermediate in enumerate(intermediates['Experiment'][intermediates['Metal'] == metal].unique()):
#                 intermediate_filter = (intermediates['Experiment'] == intermediate) & (intermediates['Measurement'] == 1)
#                 fig.add_trace(
#                     go.Scatter(
#                         x=intermediates['Energy_Corrected'][intermediate_filter], 
#                         y=intermediates['Normalized'][intermediate_filter], 
#                         mode='lines',
#                         name=intermediate,
#                         line=dict(
#                             width=3,
#                             color=px.colors.qualitative.D3[i],
#                         ),
#                         xhoverformat=x_formatting,
#                     ))
#         # Set limits of x-axis to match the edge measurements
#         fig.update_xaxes(
#             range=[np.amin(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == 1)]), 
#             np.amax(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == 1)])]
#             )
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Energy [eV]</b>',
#             yaxis_title='<b>Normalized</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#         )
#         # Customize the hover labels
#         hovertemplate = 'Normalized = %{y:.2f}'
#         fig.update_traces(hovertemplate=hovertemplate)
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_insitu_waterfall(
#     data: pd.DataFrame, 
#     experiment: str, 
#     lines: Union[list[int], None]=None,
#     vmin: Union[float, None]=None,
#     vmax: Union[float, None]=None,
#     y_axis: str='Measurement',
#     time_unit: str='seconds',
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='in-situ_waterfall.png'
# ) -> None:
#     """Waterfall plot of all normalized measurements in an in-situ experiment.

#     Reference lines can be added to highlight certain measurements or transitions in the data.

#     Args:
#         data (pd.DataFrame): The dataset.
#         experiment (str): The experiment to plot.
#         lines (optional, Union[list[int], None]): The measurment number to draw a horizontal line at. Defaults to None.
#         vmin (optional, Union[float, None]): The minimum value in the color range. If "None" the minimum value in the data is used. Defaults to None.
#         vmax (optional, Union[float, None]): The maximum value in the color range. If "None" the maximum value in the data is used. Defaults to None.
#         y_axis (optional, str): The column to use as the y-axis. Defaults to 'Measurement'.
#         time_unit (optional, str): The unit of time to use. Can be either written out (seconds) or just the first letter (s). Defaults to 'seconds'.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'in-situ_waterfall.png'.

#     Returns:
#         None
#     """
#     # Check if metal exists in the dataset
#     assert experiment in data.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {data.Experiment.unique()}'
#     # Collect the relevant columns for plotting
#     df_plot = data[['Measurement', 'Relative Time', 'Energy_Corrected', 'Normalized']][data['Experiment'] == experiment]
#     # Ensure time unit is all lowercase
#     time_unit = time_unit.lower()
#     # Set unit specific parameters
#     if (time_unit == 'seconds') or (time_unit == 's'):
#         # Set time label
#         time_label = 'Relative Time [s]'
#         # Set conversion value
#         unit_conversion = 1
#         # Set dtype
#         time_dtype = np.int32
#         # Set number of decimals
#         n_decimals = 0
#     elif (time_unit == 'minutes') or (time_unit == 'm'):
#         # Set time label
#         time_label = 'Relative Time [min]'
#         # Set conversion value
#         unit_conversion = 60
#         # Set dtype
#         time_dtype = np.float32
#         # Set number of decimals
#         n_decimals = 1
#     elif (time_unit == 'hours') or (time_unit == 'h'):
#         # Set time label
#         time_label = 'Relative Time [h]'
#         # Set conversion value
#         unit_conversion = 60 * 60
#         # Set dtype
#         time_dtype = np.float32
#         # Set number of decimals
#         n_decimals = 2
#     # Define axis specific variables
#     if y_axis == 'Measurement':
#         # Set ylabel
#         y_label = y_axis
#         # Convert y values
#         df_plot['Plotting Y'] = df_plot[y_axis]
#     elif y_axis == 'Relative Time':
#         # Set y_label 
#         y_label = time_label
#         # Convert y values
#         df_plot['Plotting Y'] = (df_plot[y_axis] / unit_conversion).astype(dtype=time_dtype)
#     # Create arrays for hover data
#     if interactive:
#         n_rows = df_plot['Measurement'].unique().shape[0]
#         measurement_array = df_plot['Measurement'].to_numpy().reshape(n_rows, -1)
#         time_array = (df_plot['Relative Time'] / unit_conversion).astype(dtype=time_dtype).to_numpy().reshape(n_rows, -1)
#     # Create pivot table of relevant data
#     heatmap_data = df_plot.pivot('Plotting Y', 'Energy_Corrected', 'Normalized')
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         ax = sns.heatmap(
#             data=heatmap_data,
#             vmin=vmin,
#             vmax=vmax,
#             cmap='viridis',
#             cbar=True,
#         )
#         # Plot horizontial lines
#         if lines != None:
#             for line in lines:
#                 position = line - 1 
#                 plt.axhline(
#                     y=position, 
#                     color='red', 
#                     linestyle='--',
#                     linewidth=1.5,
#                 )
#                 # Plot annotation text
#                 plt.annotate(
#                     text=f'Measurement {line}',
#                     xy=(plt.xlim()[0], position),
#                     xytext=(5,3),
#                     textcoords='offset points',
#                     fontsize=10,
#                     fontweight='bold',
#                     color='white',
#                 )
#         # Turn on ticks
#         plt.tick_params(
#             bottom=True,
#             left=True,
#         )
#         # Set xtick labels
#         xticks = [
#             np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]), 
#             np.amax(data['Energy_Corrected'][data['Experiment'] == experiment])
#             ] 
#         xticks += list(
#             np.arange(
#                 np.ceil(np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
#                 np.ceil(np.amax(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
#                 step=50,
#                 dtype=np.int32,
#             )
#         )
#         plt.xticks(
#             ticks=xticks - xticks[0],
#             labels=xticks,
#             rotation=0,
#         )
#         # Formatting of y-axis ticks and labels
#         # Set ytick positions
#         ytick_base = 25
#         plt.gca().yaxis.set_major_locator(ticker.IndexLocator(base=ytick_base, offset=ytick_base - 1))
#         y_pos = plt.yticks()[0].astype(int)
#         y_pos = np.append(y_pos, 0)
#         if y_axis == 'Measurement':
#             # Set ytick labels
#             plt.yticks(
#                 y_pos,
#                 data[y_axis][data['Experiment'] == experiment].unique()[y_pos],
#             )
#         elif y_axis == 'Relative Time':
#             # Converted y values
#             y_converted = (data[y_axis][data['Experiment'] == experiment].unique() / unit_conversion).astype(dtype=time_dtype)
#             # Round values
#             y_converted = np.round(
#                 y_converted,
#                 decimals=n_decimals,
#             )
#             # Set ytick labels
#             plt.yticks(
#                 ticks=y_pos,
#                 labels=y_converted[y_pos],
#             )
#         # Invert y-axis
#         plt.gca().invert_yaxis()
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Energy [eV]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             y_label, 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Format colorbar
#         cbar_ax = ax.figure.axes[-1]
#         cbar_ax.set_ylabel(
#             ylabel='Normalized',
#             size=14,
#             weight='bold'
#         )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Plot the measurements of the selected experiment/edge
#         fig = px.imshow(
#             img=heatmap_data,
#             zmin=vmin,
#             zmax=vmax,
#             origin='lower',
#             color_continuous_scale='viridis',
#             aspect='auto',
#             labels=dict(color='<b>Normalized</b>'),
#         )
#         customdata = np.stack((measurement_array, time_array), axis=-1)
#         fig.update_traces(
#             customdata=customdata,
#         )
#         # Plot horizontial lines
#         if lines != None:
#             for line in lines:
#                 position = line - 1 
#                 fig.add_hline(
#                     y=df_plot['Plotting Y'].unique()[position], 
#                     line_color='red', 
#                     line_dash='dash',
#                     line_width=1.5,
#                     annotation_text=f'<b>Measurement {line}</b>',
#                     annotation_position='top left',
#                     annotation_font=dict(
#                         color='white',
#                         size=11,
#                     )
#                 )
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Energy [eV]</b>',
#             yaxis_title=f'<b>{y_label}</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='closest',
#             coloraxis=dict(
#                 colorbar=dict(
#                     titleside='right',
#                 ),
#             ),
#         )
#         # Customize the hover labels
#         if n_decimals == 0:
#             time_format = ' = %{customdata[1]:.0f}<br>'
#         elif n_decimals == 1:
#             time_format = ' = %{customdata[1]:.1f}<br>'
#         elif n_decimals == 2:
#             time_format = ' = %{customdata[1]:.2f}<br>'
#         hovertemplate = 'Normalized = %{z:.2f}<br>' + 'Measurement = %{customdata[0]:.0f}<br>' + time_label + time_format + 'Energy [eV] = %{x:.0f}<extra></extra>'
#         fig.update_traces(hovertemplate=hovertemplate)
#         # Add and format spikes
#         fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
#         fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None
# # TODO: Make y axis relative to the reference frame
# def plot_insitu_change(
#     data: pd.DataFrame, 
#     experiment: str, 
#     reference_measurement: int=1,
#     lines: Union[list[int], None]=None,
#     vmin: Union[float, None]=None,
#     vmax: Union[float, None]=None,
#     y_axis: str='Measurement',
#     time_unit: str='seconds',
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='in-situ_change.png'
# ) -> None:
#     """Waterfall plot of the difference between all normalized measurements and a reference measurement in an in-situ experiment.

#     Reference lines can be added to highlight certain measurements or transitions in the data.

#     Args:
#         data (pd.DataFrame): The dataset.
#         experiment (str): The experiment to plot.
#         reference_measurement (int): The measurement number to use as the reference measurement. Defaults to 1.
#         lines (optional, Union[list[int], None]): The measurment number to draw a horizontal line at. Defaults to None.
#         vmin (optional, Union[float, None]): The minimum value in the color range. If "None" the minimum value in the data is used. Defaults to None.
#         vmax (optional, Union[float, None]): The maximum value in the color range. If "None" the maximum value in the data is used. Defaults to None.
#         y_axis (optional, str): The column to use as the y-axis. Defaults to 'Measurement'.
#         time_unit (optional, str): The unit of time to use. Can be either written out (seconds) or just the first letter (s). Defaults to 'seconds'.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'in-situ_change.png'.

#     Returns:
#         None
#     """
#     # Check if metal exists in the dataset
#     assert experiment in data.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {data.Experiment.unique()}'
#     # Create dataframe from subset of data
#     df_change = data[['Measurement', 'Relative Time', 'Energy_Corrected', 'Normalized']][data['Experiment'] == experiment]
#     # Extract the reference measurement
#     reference_data = df_change['Normalized'][(data['Measurement'] == reference_measurement)].to_numpy()
#     # Substract reference from all measurements
#     difference_from_reference = df_change['Normalized'].to_numpy().reshape(-1, reference_data.shape[0]) - reference_data
#     # Make new column for differences
#     df_change['ref_delta'] = difference_from_reference.reshape(-1)
#     # Ensure time unit is all lowercase
#     time_unit = time_unit.lower()
#     # Set unit specific parameters
#     if (time_unit == 'seconds') or (time_unit == 's'):
#         # Set time label
#         time_label = 'Relative Time [s]'
#         # Set conversion value
#         unit_conversion = 1
#         # Set dtype
#         time_dtype = np.int32
#         # Set number of decimals
#         n_decimals = 0
#     elif (time_unit == 'minutes') or (time_unit == 'm'):
#         # Set time label
#         time_label = 'Relative Time [min]'
#         # Set conversion value
#         unit_conversion = 60
#         # Set dtype
#         time_dtype = np.float32
#         # Set number of decimals
#         n_decimals = 1
#     elif (time_unit == 'hours') or (time_unit == 'h'):
#         # Set time label
#         time_label = 'Relative Time [h]'
#         # Set conversion value
#         unit_conversion = 60 * 60
#         # Set dtype
#         time_dtype = np.float32
#         # Set number of decimals
#         n_decimals = 2
#     # Define axis specific variables
#     if y_axis == 'Measurement':
#         # Set ylabel
#         y_label = y_axis
#         # Convert y values
#         df_change['Plotting Y'] = df_change[y_axis]
#     elif y_axis == 'Relative Time':
#         # Set y_label 
#         y_label = time_label
#         # Convert y values
#         df_change['Plotting Y'] = (df_change[y_axis] / unit_conversion).astype(dtype=time_dtype)
#     # Create arrays for hover data
#     if interactive:
#         n_rows = df_change['Measurement'].unique().shape[0]
#         measurement_array = df_change['Measurement'].to_numpy().reshape(n_rows, -1)
#         time_array = (df_change['Relative Time'] / unit_conversion).astype(dtype=time_dtype).to_numpy().reshape(n_rows, -1)
#     # Create pivot table of relevant data
#     heatmap_data = df_change.pivot('Plotting Y', 'Energy_Corrected', 'ref_delta')
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         ax = sns.heatmap(
#             data=heatmap_data,
#             center=0.,
#             vmin=vmin,
#             vmax=vmax,
#             cmap='seismic',
#             cbar=True,
#         )
#         # Plot reference line 
#         plt.axhline(
#             y=reference_measurement - 1, 
#             color='k', 
#             linestyle='-',
#             linewidth=1.5,
#         )
#         # Plot reference annotation
#         plt.annotate(
#             text=f'Reference ({reference_measurement})',
#             xy=(plt.xlim()[0], reference_measurement-1),
#             xytext=(5,3),
#             textcoords='offset points',
#             fontsize=10,
#             fontweight='bold',
#             color='k',
#         )
#         # Plot horizontial lines
#         if lines != None:
#             for line in lines:
#                 position = line - 1
#                 plt.axhline(
#                     y=position, 
#                     color='k', 
#                     linestyle='--',
#                     linewidth=1.5,
#                 )
#                 # Plot annotation text
#                 plt.annotate(
#                     text=f'Measurement {line}',
#                     xy=(plt.xlim()[0], position),
#                     xytext=(5,3),
#                     textcoords='offset points',
#                     fontsize=10,
#                     fontweight='bold',
#                     color='k',
#                 )
#         # Turn on ticks
#         plt.tick_params(
#             bottom=True,
#             left=True,
#         )
#         # Set xtick labels
#         xticks = [
#             np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]), 
#             np.amax(data['Energy_Corrected'][data['Experiment'] == experiment])
#             ] 
#         xticks += list(
#             np.arange(
#                 np.ceil(np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
#                 np.ceil(np.amax(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
#                 step=50,
#                 dtype=np.int32,
#             )
#         )
#         plt.xticks(
#             ticks=xticks - xticks[0],
#             labels=xticks,
#             rotation=0,
#         )
#         # Formatting of y-axis ticks and labels
#         # Set ytick positions
#         ytick_base = 25
#         plt.gca().yaxis.set_major_locator(ticker.IndexLocator(base=ytick_base, offset=ytick_base - 1))
#         y_pos = plt.yticks()[0].astype(int)
#         y_pos = np.append(y_pos, 0)
#         if y_axis == 'Measurement':
#             # Set ytick labels
#             plt.yticks(
#                 y_pos,
#                 data[y_axis][data['Experiment'] == experiment].unique()[y_pos],
#             )
#         elif y_axis == 'Relative Time':
#             # Converted y values
#             y_converted = (data[y_axis][data['Experiment'] == experiment].unique() / unit_conversion).astype(dtype=time_dtype)
#             # Round values
#             y_converted = np.round(
#                 y_converted,
#                 decimals=n_decimals,
#             )
#             # Set ytick labels
#             plt.yticks(
#                 ticks=y_pos,
#                 labels=y_converted[y_pos],
#             )
#         # Invert y-axis
#         plt.gca().invert_yaxis()
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Energy [eV]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             y_label, 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Format colorbar
#         cbar_ax = ax.figure.axes[-1]
#         cbar_ax.set_ylabel(
#             ylabel=r'$\mathbf{\Delta}$ Normalized intensity',
#             size=14,
#             weight='bold'
#         )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Plot the measurements of the selected experiment/edge
#         fig = px.imshow(
#             img=heatmap_data,
#             zmin=vmin,
#             zmax=vmax,
#             origin='lower',
#             color_continuous_scale='RdBu_r',
#             color_continuous_midpoint=0.,
#             aspect='auto',
#             labels=dict(color='<b>\u0394 Normalized intensity</b>'),
#         )
#         customdata = np.stack((measurement_array, time_array), axis=-1)
#         fig.update_traces(
#             customdata=customdata,
#         )
#         fig.add_hline(
#             y=df_change['Plotting Y'].unique()[reference_measurement - 1], 
#             line_color='black', 
#             line_width=1.5,
#             annotation_text=f'<b>Reference ({reference_measurement})</b>',
#             annotation_position='top left',
#             annotation_font=dict(
#                 color='black',
#                 size=11,
#             )
#         )
#         # Plot horizontial lines
#         if lines != None:
#             for line in lines:
#                 position = line - 1 
#                 fig.add_hline(
#                     y=df_change['Plotting Y'].unique()[position], 
#                     line_color='black', 
#                     line_dash='dash',
#                     line_width=1.5,
#                     annotation_text=f'<b>Measurement {line}</b>',
#                     annotation_position='top left',
#                     annotation_font=dict(
#                         color='black',
#                         size=11,
#                     )
#                 )
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Energy [eV]</b>',
#             yaxis_title=f'<b>{y_label}</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='closest',
#             coloraxis=dict(
#                 colorbar=dict(
#                     titleside='right',
#                 ),
#             ),
#         )
#         # Customize the hover labels
#         if n_decimals == 0:
#             time_format = ' = %{customdata[1]:.0f}<br>'
#         elif n_decimals == 1:
#             time_format = ' = %{customdata[1]:.1f}<br>'
#         elif n_decimals == 2:
#             time_format = ' = %{customdata[1]:.2f}<br>'
#         hovertemplate = '\u0394 Normalized intensity = %{z:.2f}<br>' + 'Measurement = %{customdata[0]:.0f}<br>' + time_label + time_format + 'Energy [eV] = %{x:.0f}<extra></extra>'
#         fig.update_traces(hovertemplate=hovertemplate)
#         # Add and format spikes
#         fig.update_xaxes(showspikes=True, spikecolor="black", spikethickness=-2)
#         fig.update_yaxes(showspikes=True, spikecolor="black", spikethickness=-2)
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_temperatures(
#     df: pd.DataFrame, 
#     with_uncertainty: bool=True, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='temperature_curves.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting visual comparison of when the different metals in the experiment reduce by showing the weight of the metal foil component determined from Linear Combination Analysis (LCA).

#     Args:
#         df (pd.DataFrame): Results of LCA.
#         with_uncertainty (optional, bool): Boolean flag controlling if the uncertainty on the average emperature is plotted. Defaults to True.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'temperature_curves.png'.

#     Returns:
#         None
#     """  
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Plot temperature curves
#         sns.lineplot(
#             data=df,
#             x='Measurement',
#             y='Temperature',
#             hue='Metal',
#             ci=None,
#             linewidth=2,
#             palette='colorblind',
#         )
#         # Create filter for relevant values
#         avg_filter = (df['Parameter'] == 'foil_weight') & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & (df['Metal'] == df['Metal'].unique().tolist()[0])
#         # Plot temperature average
#         sns.lineplot(
#             data=df[avg_filter],
#             x='Measurement',
#             y='Temperature Average',
#             ci=None,
#             linewidth=3,
#             color='k',
#             label='Average',
#         )
#         # Plot the uncertainty on the average temperature
#         if with_uncertainty:
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             plt.fill_between(
#                 df['Measurement'][avg_filter], 
#                 df['Temperature Average'][avg_filter] - df['Temperature Std'][avg_filter], 
#                 df['Temperature Average'][avg_filter] + df['Temperature Std'][avg_filter], 
#                 alpha=0.3,
#                 color='k',
#                 )
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Measurement', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             'Temperature [°C]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend(
#             title='Metal',
#             fontsize=12,
#             title_fontsize=13,
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Formatting of the hover label "title"
#         x_formatting = '.0f'
#         # Create filter for relevant values
#         df_filter = (df['Parameter'] == 'foil_weight') & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0])
#         # Plot the measurements of the selected experiment/edge
#         fig = px.line(
#             data_frame=df[df_filter],
#             x='Measurement',
#             y='Temperature',
#             color='Metal',
#             color_discrete_sequence=sns.color_palette('colorblind').as_hex()
#         )
#         # Change line formatting
#         fig.update_traces(
#             line=dict(
#                 width=2,
#             ),
#             xhoverformat=x_formatting,
#         )
#         # Customize the hover labels
#         hovertemplate = 'Temperature = %{y:.1f} °C'
#         fig.update_traces(hovertemplate=hovertemplate)
#         if with_uncertainty:
#             # Change hover template for line with uncertainty
#             hovertemplate = 'Temperature = %{y:.1f} +/- %{customdata:.1f} °C'
#         # Create filter for relevant values
#         avg_filter = df_filter & (df['Metal'] == df['Metal'].unique().tolist()[0])
#         # Plot temperature average
#         fig.add_trace(
#             go.Scatter(
#                 x=df['Measurement'][avg_filter], 
#                 y=df['Temperature Average'][avg_filter], 
#                 mode='lines',
#                 name='Average',
#                 legendgroup='Average',
#                 line=dict(
#                     width=3,
#                     color='black',
#                 ),
#                 xhoverformat=x_formatting,
#                 customdata=df['Temperature Std'][avg_filter],
#                 hovertemplate=hovertemplate,
#             ))
#         # Plot the uncertainty on the average temperature
#         if with_uncertainty:
#             # Create lists with the x-values and upper/lower error bounds
#             x_range = df['Measurement'][avg_filter].tolist()
#             std_upper = (df['Temperature Average'][avg_filter] + df['Temperature Std'][avg_filter]).tolist()
#             std_lower = (df['Temperature Average'][avg_filter] - df['Temperature Std'][avg_filter]).tolist()
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             fig.add_trace(
#                 go.Scatter(
#                     x=x_range + x_range[::-1], 
#                     y=std_upper + std_lower[::-1], 
#                     fill='toself',
#                     fillcolor='rgba(0,0,0,0.3)',
#                     line=dict(color='rgba(0,0,0,0)'),
#                     hoverinfo="skip",
#                     showlegend=False,
#                     legendgroup='Average',
#                     xhoverformat=x_formatting,
#                 ))
#         # Set limits of x-axis to match the edge measurements
#         fig.update_xaxes(
#             range=[np.amin(df['Measurement'][avg_filter]), 
#             np.amax(df['Measurement'][avg_filter])]
#             )
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Measurement</b>',
#             yaxis_title='<b>Temperature [°C]</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#         )
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_LCA(
#     results: pd.DataFrame, 
#     data: pd.DataFrame, 
#     experiment: str, 
#     measurement: int, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='LCA_plot.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting of the result of Linear Combination Analysis (LCA) for a single measurement and combination of metal and precursor.

#     Args:
#         results (pd.DataFrame): Results from LCA.
#         data (pd.DataFrame): Dataset.
#         experiment (str): Combination of metal and precursor to plot. Given in the format '{metal} + {precursor}'.
#         measurement (int): Measurement to plot.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'LCA_plot.png'.

#     Returns:
#         None
#     """    
#     # Check if experiment exists in dataset
#     assert experiment in results.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {results.Experiment.unique()}'
#     # Check if measurement exists in dataset
#     assert measurement in results["Measurement"][results["Experiment"] == experiment].unique(), f'No measurement with the name: {measurement}\n\nValid values are: {results["Measurement"][results["Experiment"] == experiment].unique()}'
#     # Extract the metal (, intermediate) and precursor from the experiment
#     components = experiment.split(' + ')
#     if len(components) == 2:
#         precursor, product = components
#         intermediate = None
#     elif len(components) == 3:
#         precursor, intermediate, product = components
#     # Create filters for relevant values
#     data_filter = (data['Experiment'] == data['Experiment'][data['Metal'] == results['Metal'][results['Experiment'] == experiment].unique()[0]].unique()[0]) & (data['Measurement'] == measurement)
#     product_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'product_weight') & (results['Measurement'] == measurement)
#     precursor_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'precursor_weight') & (results['Measurement'] == measurement)
#     if intermediate != None:
#         intermediate_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'intermediate_weight') & (results['Measurement'] == measurement)
#     # Scale the basis functions with their component weight
#     product_component = (results['Value'][product_filter].to_numpy() * results['Basis Function'][product_filter].to_numpy())[0]
#     precursor_component = (results['Value'][precursor_filter].to_numpy() * results['Basis Function'][precursor_filter].to_numpy())[0]
#     if intermediate != None:
#         intermediate_component = (results['Value'][intermediate_filter].to_numpy() * results['Basis Function'][intermediate_filter].to_numpy())[0]
#         component_sum = precursor_component + intermediate_component + product_component
#     else:
#         component_sum = precursor_component + product_component
#     # Define colors to use
#     color_list = sns.color_palette('colorblind').as_hex()    
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Plot reference line at 0
#         plt.axhline(y=0, c='k', alpha=0.5, lw=3, ls='--')
#         # Plot measurement
#         sns.lineplot(
#             data=data[data_filter], 
#             x='Energy_Corrected', 
#             y='Normalized', 
#             color='k',
#             linewidth=3,
#             label=f'Measurement {measurement}',
#             )
#         # Plot LCA approximation
#         sns.lineplot( 
#             x=results['Energy Range'][product_filter].to_numpy()[0], 
#             y=component_sum, 
#             color='cyan',
#             linewidth=3,
#             label='LCA approx.'
#             )
#         # Plot product component
#         sns.lineplot( 
#             x=results['Energy Range'][product_filter].to_numpy()[0], 
#             y=product_component, 
#             color=color_list.pop(0),
#             linewidth=3,
#             label=f'Product ({results["Value"][product_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][product_filter].to_numpy()[0]:.1%})'
#             )
#         if intermediate != None:
#             # Plot intermediate component
#             sns.lineplot( 
#                 x=results['Energy Range'][intermediate_filter].to_numpy()[0], 
#                 y=intermediate_component, 
#                 color=color_list.pop(0),
#                 linewidth=3,
#                 label=f'Intermediate ({results["Value"][intermediate_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][intermediate_filter].to_numpy()[0]:.1%})'
#                 )
#         # Plot precursor component
#         sns.lineplot( 
#             x=results['Energy Range'][precursor_filter].to_numpy()[0], 
#             y=precursor_component, 
#             color=color_list.pop(0),
#             linewidth=3,
#             label=f'Precursor ({results["Value"][precursor_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][precursor_filter].to_numpy()[0]:.1%})'
#             )
#         # Plot LCA residual
#         sns.lineplot( 
#             x=results['Energy Range'][product_filter].to_numpy()[0], 
#             y=data['Normalized'][data_filter & data['Energy_Corrected'].isin(results['Energy Range'][product_filter].to_numpy()[0])] - component_sum, 
#             color='r',
#             linewidth=3,
#             label='Residual'
#             )
#         # Ensure the y-axis covers atleast the range from 0 - 1
#         y_lim_bot, y_lim_top = plt.ylim()
#         if y_lim_bot > 0:
#             y_lim_bot = 0
#         if y_lim_top < 1:
#             y_lim_top = 1
#         # Ensure the y-axis isn't outside -0.5 - 1.5
#         if y_lim_bot < -0.5:
#             y_lim_bot = -0.5
#         if y_lim_top > 1.5:
#             y_lim_top = 1.5
#         plt.ylim((y_lim_bot, y_lim_top))
#         # Set limits of x-axis to match the edge measurements
#         plt.xlim(
#             (np.amin(data['Energy_Corrected'][data_filter]),
#             np.amax(data['Energy_Corrected'][data_filter]))
#             )
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             'Energy [eV]', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             'Normalized X-ray transmission', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend( 
#             title=f'{data["Metal"][data_filter].unique()[0]} edge', 
#             fontsize=12, 
#             title_fontproperties=dict(size=14, weight='bold'),
#             ncol=3,
#             loc='lower center',
#             bbox_to_anchor=(0.5, 1.01)
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Formatting of the hover label "title"
#         x_formatting = '.0f'
#         # Plot the measurements of the selected experiment/edge
#         fig = go.Figure(data=go.Scatter(
#             x=data['Energy_Corrected'][data_filter],
#             y=data['Normalized'][data_filter],
#             mode='lines',
#             name=f'Measurement {measurement}',
#             line=dict(
#                 width=3,
#                 color='black',
#             ),
#             xhoverformat=x_formatting,
#         ))
#         # Plot reference line at 0
#         fig.add_hline(
#             y=0,
#             line_width=3,
#             line_dash='dash',
#             line_color='rgba(0,0,0,0.5)'
#         )
#         # Plot the LCA approximation
#         fig.add_trace(
#             go.Scatter(
#                 x=results['Energy Range'][product_filter].to_numpy()[0], 
#                 y=component_sum, 
#                 mode='lines',
#                 name='LCA approx.',
#                 line=dict(
#                     width=3,
#                     color='magenta',
#                 ),
#                 xhoverformat=x_formatting,
#             ))
#         # Plot the LCA product component
#         fig.add_trace(
#             go.Scatter(
#                 x=results['Energy Range'][product_filter].to_numpy()[0], 
#                 y=product_component, 
#                 mode='lines',
#                 name=f'Product ({results["Value"][product_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][product_filter].to_numpy()[0]:.1%})',
#                 line=dict(
#                     width=3,
#                     color=color_list.pop(0),
#                 ),
#                 xhoverformat=x_formatting,
#             ))
#         if intermediate != None:
#             # Plot the LCA intermediate component
#             fig.add_trace(
#                 go.Scatter(
#                     x=results['Energy Range'][intermediate_filter].to_numpy()[0], 
#                     y=intermediate_component, 
#                     mode='lines',
#                     name=f'Intermediate ({results["Value"][intermediate_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][intermediate_filter].to_numpy()[0]:.1%})',
#                     line=dict(
#                         width=3,
#                         color=color_list.pop(0),
#                     ),
#                     xhoverformat=x_formatting,
#                 ))
#         # Plot the LCA precursor component
#         fig.add_trace(
#             go.Scatter(
#                 x=results['Energy Range'][precursor_filter].to_numpy()[0], 
#                 y=precursor_component, 
#                 mode='lines',
#                 name=f'Precursor ({results["Value"][precursor_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][precursor_filter].to_numpy()[0]:.1%})',
#                 line=dict(
#                     width=3,
#                     color=color_list.pop(0),
#                 ),
#                 xhoverformat=x_formatting,
#             ))
#         # Plot the LCA residual
#         fig.add_trace(
#             go.Scatter(
#                 x=results['Energy Range'][product_filter].to_numpy()[0], 
#                 y=data['Normalized'][data_filter & data['Energy_Corrected'].isin(results['Energy Range'][product_filter].to_numpy()[0])] - component_sum, 
#                 mode='lines',
#                 name='Residual',
#                 line=dict(
#                     width=3,
#                     color='red',
#                 ),
#                 xhoverformat=x_formatting,
#             ))
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title='<b>Energy [eV]</b>',
#             yaxis_title='<b>Normalized</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#         )
#         # Customize the hover labels
#         hovertemplate = 'Normalized = %{y:.2f}'
#         fig.update_traces(hovertemplate=hovertemplate)
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_LCA_change(
#     df: pd.DataFrame, 
#     product: str, 
#     precursor: str, 
#     intermediate: Union[str, None]=None,
#     x_axis: str='Measurement', 
#     with_uncertainty: bool=True, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='LCA_change.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting the change in weights determined from Linear Combination Analysis (LCA) over an entire experiment.

#     Args:
#         df (pd.DataFrame): Results from LCA.
#         product (str): The product that should be plotted.
#         precursor (str): The precursor that should be plotted.
#         intermediate (optional, Union[str, None]): The intermediate that should be plotted. Defaults to None.
#         x_axis (optional, str): The column to plot on the a-axis. Defaults to 'Measurement'.
#         with_uncertainty (optional, bool): Boolean flag controlling if the uncertainties on the weights are plotted. Defaults to True.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'LCA_change.png'.

#     Returns:
#         None
#     """    
#     # Check if metal exists in dataset
#     assert product in df.Product.unique(), f'No product with the name: {product}\n\nValid values are: {df.Product.unique()}'
#     # Check if precursor exists in dataset
#     assert precursor in df.Precursor.unique(), f'No precursor with the name: {precursor}\n\nValid values are: {df.Precursor.unique()}'
#     # Check if intermediate exists in dataset
#     assert intermediate in df.Intermediate.unique(), f'No intermediate with the name: {intermediate}\n\nValid values are: {df.Intermediate.unique()}'
#     # Check if x_axis exists in dataset
#     assert x_axis in df.columns, f'No column with the name: {x_axis}\n\nValid values are: {df.columns}'
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Define colors to use
#         color_list = sns.color_palette('colorblind').as_hex() 
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Create filter for relevant values
#         df_filter = (df['Product'] == product) & (df['Precursor'] == precursor) & (df['Intermediate'] == intermediate)
#         # Plot weights for the two components for each measurement
#         sns.lineplot(
#             data=df[df_filter],
#             x=x_axis, 
#             y='Value', 
#             hue='Parameter', 
#             ci=None,
#             linewidth=3,
#             palette='colorblind',
#         )
#         # TODO: Make proper legend labels
#         # Plot the uncertainty on the weights
#         if with_uncertainty:
#             # Create filter for relevant values
#             product_filter = (df['Parameter'] == 'product_weight') & df_filter
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             plt.fill_between(
#                 df[x_axis][product_filter], 
#                 df['Value'][product_filter] - df['StdCorrected'][product_filter], 
#                 df['Value'][product_filter] + df['StdCorrected'][product_filter], 
#                 alpha=0.3,
#                 color=color_list.pop(0),
#             )
#             if intermediate != None:
#                 # Create filter for relevant values
#                 intermediate_filter = (df['Parameter'] == 'intermediate_weight') & df_filter
#                 # Plot uncertainty as the values within +/- 1 standard deviation
#                 plt.fill_between(
#                     df[x_axis][intermediate_filter], 
#                     df['Value'][intermediate_filter] - df['StdCorrected'][intermediate_filter], 
#                     df['Value'][intermediate_filter] + df['StdCorrected'][intermediate_filter], 
#                     alpha=0.3,
#                     color=color_list.pop(0),
#                 )
#             # Create filter for relevant values
#             precursor_filter = (df['Parameter'] == 'precursor_weight') & df_filter
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             plt.fill_between(
#                 df[x_axis][precursor_filter], 
#                 df['Value'][precursor_filter] - df['StdCorrected'][precursor_filter], 
#                 df['Value'][precursor_filter] + df['StdCorrected'][precursor_filter], 
#                 alpha=0.3,
#                 color=color_list.pop(0),
#             )
#         # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
#         y_min, y_max = plt.ylim()
#         plt.ylim(
#             (np.amax([y_min, -0.5]), np.amin([y_max, 1.5]))
#         )
#         # Specify the units used on the x-axis
#         if 'Temperature' in x_axis:
#             units = ' [°C]'
#         else:
#             units = ''
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             x_axis + units, 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             'Weight fraction', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend(
#             labels=[product, precursor, intermediate], 
#             title='Components',
#             fontsize=12,
#             title_fontsize=13,
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Define colors to use
#         color_list = sns.color_palette('colorblind') 
#         # Formatting of the hover label "title"
#         x_formatting = '.1f'
#         # Variables to catch min and max axis values
#         y_min = -0.02
#         y_max = 1.02
#         # Create filter for relevant values
#         df_filter = (df['Product'] == product) & (df['Precursor'] == precursor) #& (df['Intermediate'] == intermediate)
#         # Plot the LCA weights over time
#         fig = px.line(
#             data_frame=df[df_filter],
#             x=x_axis,
#             y='Value',
#             color='Parameter',
#             color_discrete_sequence=color_list.as_hex(),
#             custom_data=['StdCorrected'],
#         )
#         # Change line formatting
#         fig.update_traces(
#             line=dict(
#                 width=3,
#             ),
#             xhoverformat=x_formatting,
#         )
#         if with_uncertainty:
#             # Customize the hover labels
#             hovertemplate = 'Weight = %{y:.2f} +/- %{customdata[0]:.2f}'
#             fig.update_traces(hovertemplate=hovertemplate)
#         else:
#             # Customize the hover labels
#             hovertemplate = 'Weight = %{y:.2f}'
#             fig.update_traces(hovertemplate=hovertemplate)
#         # Plot the uncertainty on the weights
#         if with_uncertainty:
#             # Create filter for relevant values
#             product_filter = (df['Parameter'] == 'product_weight') & df_filter
#             # Create lists with the x-values and upper/lower error bounds
#             x_range = df[x_axis][product_filter].tolist()
#             std_upper = (df['Value'][product_filter] + df['StdCorrected'][product_filter]).tolist()
#             std_lower = (df['Value'][product_filter] - df['StdCorrected'][product_filter]).tolist()
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             fig.add_trace(
#                 go.Scatter(
#                     x=x_range + x_range[::-1], 
#                     y=std_upper + std_lower[::-1], 
#                     fill='toself',
#                     fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
#                     line=dict(color='rgba(0,0,0,0)'),
#                     hoverinfo="skip",
#                     showlegend=False,
#                     legendgroup='product_weight',
#                     xhoverformat=x_formatting,
#                 ))
#             if intermediate != None:
#                 # Create filter for relevant values
#                 intermediate_filter = (df['Parameter'] == 'intermediate_weight') & df_filter
#                 # Create lists with the x-values and upper/lower error bounds
#                 x_range = df[x_axis][intermediate_filter].tolist()
#                 std_upper = (df['Value'][intermediate_filter] + df['StdCorrected'][intermediate_filter]).tolist()
#                 std_lower = (df['Value'][intermediate_filter] - df['StdCorrected'][intermediate_filter]).tolist()
#                 # Plot uncertainty as the values within +/- 1 standard deviation
#                 fig.add_trace(
#                     go.Scatter(
#                         x=x_range + x_range[::-1], 
#                         y=std_upper + std_lower[::-1], 
#                         fill='toself',
#                         fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
#                         line=dict(color='rgba(0,0,0,0)'),
#                         hoverinfo="skip",
#                         showlegend=False,
#                         legendgroup='intermediate_weight',
#                         xhoverformat=x_formatting,
#                     ))  
#             # Create filter for relevant values
#             precursor_filter = (df['Parameter'] == 'precursor_weight') & df_filter
#             # Create lists with the x-values and upper/lower error bounds
#             x_range = df[x_axis][precursor_filter].tolist()
#             std_upper = (df['Value'][precursor_filter] + df['StdCorrected'][precursor_filter]).tolist()
#             std_lower = (df['Value'][precursor_filter] - df['StdCorrected'][precursor_filter]).tolist()
#             # Plot uncertainty as the values within +/- 1 standard deviation
#             fig.add_trace(
#                 go.Scatter(
#                     x=x_range + x_range[::-1], 
#                     y=std_upper + std_lower[::-1], 
#                     fill='toself',
#                     fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
#                     line=dict(color='rgba(0,0,0,0)'),
#                     hoverinfo="skip",
#                     showlegend=False,
#                     legendgroup='precursor_weight',
#                     xhoverformat=x_formatting,
#                 ))  
#             # Check if largest value is above current axis range
#             if y_max < np.amax(std_upper):
#                 y_max = np.amax(std_upper)
#             # Check if smallest values is below current axis range
#             if y_min > np.amin(std_lower):
#                 y_min = np.amin(std_lower)
#         # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
#         fig.update_yaxes(
#             range=[np.amax([y_min, -0.5]), np.amin([y_max, 1.5])]
#         )
#         # Set limits of x-axis to match the edge measurements
#         fig.update_xaxes(
#             range=[np.amin(df[x_axis][df_filter]), 
#             np.amax(df[x_axis][df_filter])]
#             )
#         # Specify the units used on the x-axis
#         if 'Temperature' in x_axis:
#             units = ' [°C]'
#         else:
#             units = ''
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title=f'<b>{x_axis}{units}</b>',
#             yaxis_title='<b>Weight fraction</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#         )
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# def plot_reduction_comparison(
#     df: pd.DataFrame, 
#     precursor_type: str='all', 
#     x_axis: str='Measurement', 
#     with_uncertainty: bool=True, 
#     interactive: bool=False, 
#     save_plot: bool=False, 
#     save_name: str='reduction_comparison.png'
# ) -> None:
#     """------------------------------------------------------------------
#     Plotting visual comparison of when the different metals in the experiment reduce by showing the weight of the metal foil component determined from Linear Combination Analysis (LCA).

#     Args:
#         df (pd.DataFrame): Results of LCA.
#         precursor_type (optional, str): The type of precursors to be plotted. Defaults to 'all'.
#         x_axis (optional, str): The column to plot on the a-axis. Defaults to 'Measurement'.
#         with_uncertainty (optional, bool): Boolean flag controlling if the uncertainties on the weights are plotted. Defaults to True.
#         interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
#         save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
#         save_name (optional, str): The filename of the saved plot. Defaults to 'reduction_comparison.png'.

#     Returns:
#         None
#     """    
#     # Check if x_axis exists in dataset
#     assert x_axis in df.columns, f'No column with the name: {x_axis}\n\nValid values are: {df.columns}'
#     # Make figure using matplotlib and seaborn
#     if not interactive:
#         # Create figure object and set the figure size
#         plt.figure(figsize=(8,8))
#         # Plot the weight of the foil component for all metal + precursor combinations
#         if precursor_type == 'all':
#             df_filter = (df['Parameter'] == 'foil_weight')
#             sns.lineplot(
#                 data=df[df_filter], 
#                 x=x_axis,
#                 y='Value',
#                 hue='Metal',
#                 style='Precursor Type',
#                 ci=None,
#                 linewidth=2,
#                 palette='colorblind',
#                 )
#         # Plot the weight of the foil component for all experiments with the specified type of precursor
#         else:
#             # Check if the precursor type exists in the dataset
#             assert precursor_type in df['Precursor Type'].unique(), f'No precursor type with the name: {precursor_type}\n\nValid values are: {df["Precursor Type"].unique()}'
#             # Create filter for relevant values
#             df_filter = (df['Parameter'] == 'foil_weight') & (df['Precursor'].str.contains(precursor_type))
#             sns.lineplot(
#                 data=df[df_filter], 
#                 x=x_axis,
#                 y='Value',
#                 hue='Metal',
#                 ci=None,
#                 linewidth=2,
#                 palette='colorblind',
#                 )
#         # Plot uncertainties on the foil weights
#         if with_uncertainty:
#             # Loop over each metal
#             for i, metal in enumerate(df['Metal'][df_filter].unique()):
#                 # If all metal + precursor combinations are plotted, only show uncertainties for the most common precursor type to avoid visual clutter
#                 if precursor_type == 'all':
#                     # Create filter for relevant values
#                     foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & df_filter
#                 # If a precursor type is specified, only plot uncertainties for that type of precursor
#                 else:
#                     # Create filter for relevant values
#                     foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type) & df_filter
#                 # Plot uncertainties
#                 plt.fill_between(
#                     df[x_axis][foil_filter], 
#                     df['Value'][foil_filter] - df['StdCorrected'][foil_filter], 
#                     df['Value'][foil_filter] + df['StdCorrected'][foil_filter], 
#                     alpha=0.3,
#                     color=sns.color_palette('colorblind').as_hex()[i],
#                     )
#         # Ensure the y-axis covers atleast the range from 0 - 1
#         y_lim_bot, y_lim_top = plt.ylim()
#         if y_lim_bot > 0:
#             y_lim_bot = 0
#         if y_lim_top < 1:
#             y_lim_top = 1
#         # Ensure the y-axis isn't outside -0.5 - 1.5
#         if y_lim_bot < -0.5:
#             y_lim_bot = -0.5
#         if y_lim_top > 1.5:
#             y_lim_top = 1.5
#         plt.ylim((y_lim_bot, y_lim_top))
#         # Specify the units used on the x-axis
#         if 'Temperature' in x_axis:
#             units = ' [°C]'
#         else:
#             units = ''
#         # Specify text and formatting of axis labels
#         plt.xlabel(
#             x_axis + units, 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         plt.ylabel(
#             'Weight fraction', 
#             fontsize=14, 
#             fontweight='bold'
#             )
#         # Specify placement, formatting and title of the legend
#         plt.legend(
#             labels=df['Experiment'][df_filter].unique(), 
#             title='Components',
#             fontsize=12,
#             title_fontsize=13,
#             )
#         # Enforce matplotlibs tight layout
#         plt.tight_layout()
#         # Save plot as a png
#         if save_plot:
#             plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
#         plt.show()
#     # Make interactive plot using plotly
#     elif interactive:
#         # Formatting of the hover label "title"
#         x_formatting = '.1f'
#         # Variables to catch min and max axis values
#         y_min = -0.02
#         y_max = 1.02
#         fig = go.Figure()
#         # Plot the weight of the foil component for all metal + precursor combinations
#         if precursor_type == 'all':
#             df_filter = (df['Parameter'] == 'foil_weight')
#             # Plot the LCA weights over time
#             for i, metal in enumerate(df['Metal'][df_filter].unique()):
#                 # Loop over each relevant precursor for each metal
#                 for precursor_type in df['Precursor Type'][df_filter & (df['Metal'] == metal)].unique():
#                     # Create filter for relevant values
#                     foil_filter = df_filter & (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type)
#                     # Determine linestyle based on most common precursor type
#                     if precursor_type == df['Precursor Type'].mode().to_list()[0]:
#                         linestyle = 'solid'
#                         # Used to link line and the corresponding uncertainty
#                         legend_group = f'{metal}'
#                     else:
#                         linestyle = 'dash'
#                         legend_group = None
#                     # Plot the trace
#                     fig.add_trace(
#                         go.Scatter(
#                             x=df[x_axis][foil_filter], 
#                             y=df['Value'][foil_filter], 
#                             mode='lines',
#                             name=f'{metal} + {precursor_type}',
#                             line_color=f'rgba{sns.color_palette("colorblind")[i]}',
#                             legendgroup=legend_group,
#                             line_dash=linestyle,
#                             line_width=2,
#                             customdata=df['StdCorrected'][foil_filter],
#                             xhoverformat=x_formatting,
#                         ))
#         # Plot the weight of the foil component for all experiments with the specified type of precursor
#         else:
#             # Check if the precursor type exists in the dataset
#             assert precursor_type in df['Precursor Type'].unique(), f'No precursor type with the name: {precursor_type}\n\nValid values are: {df["Precursor Type"].unique()}'
#             # Create filter for relevant values
#             df_filter = (df['Parameter'] == 'foil_weight') & (df['Precursor'].str.contains(precursor_type))
#             # Plot the LCA weights over time
#             for i, metal in enumerate(df['Metal'][df_filter].unique()):
#                 # Create filter for relevant values
#                 foil_filter = df_filter & (df['Metal'] == metal)
#                 linestyle = 'solid'
#                 # Used to link line and the corresponding uncertainty
#                 legend_group = f'{metal}'
#                 # Plot the trace
#                 fig.add_trace(
#                     go.Scatter(
#                         x=df[x_axis][foil_filter], 
#                         y=df['Value'][foil_filter], 
#                         mode='lines',
#                         name=f'{metal} + {precursor_type}',
#                         line_color=f'rgba{sns.color_palette("colorblind")[i]}',
#                         legendgroup=legend_group,
#                         line_dash=linestyle,
#                         line_width=2,
#                         customdata=df['StdCorrected'][foil_filter],
#                         xhoverformat=x_formatting,
#                     ))
#         if with_uncertainty:
#             # Customize the hover labels
#             hovertemplate = 'Weight = %{y:.2f} +/- %{customdata:.2f}'
#             fig.update_traces(hovertemplate=hovertemplate)
#         else:
#             # Customize the hover labels
#             hovertemplate = 'Weight = %{y:.2f}'
#             fig.update_traces(hovertemplate=hovertemplate)
#         # Plot uncertainties on the foil weights
#         if with_uncertainty:
#             # Loop over each metal
#             for i, metal in enumerate(df['Metal'][df_filter].unique()):
#                 # If all metal + precursor combinations are plotted, only show uncertainties for the most common precursor type to avoid visual clutter
#                 if precursor_type == 'all':
#                     # Create filter for relevant values
#                     foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & df_filter
#                 # If a precursor type is specified, only plot uncertainties for that type of precursor
#                 else:
#                     # Create filter for relevant values
#                     foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type) & df_filter
#                 # Create lists with the x-values and upper/lower error bounds
#                 x_range = df[x_axis][foil_filter].tolist()
#                 std_upper = (df['Value'][foil_filter] + df['StdCorrected'][foil_filter]).tolist()
#                 std_lower = (df['Value'][foil_filter] - df['StdCorrected'][foil_filter]).tolist()
#                 # Check if largest value is above current axis range
#                 if y_max < np.amax(std_upper):
#                     y_max = np.amax(std_upper)
#                 # Check if smallest values is below current axis range
#                 if y_min > np.amin(std_lower):
#                     y_min = np.amin(std_lower)
#                 # Plot uncertainty as the values within +/- 1 standard deviation
#                 fig.add_trace(
#                     go.Scatter(
#                         x=x_range + x_range[::-1], 
#                         y=std_upper + std_lower[::-1], 
#                         fill='toself',
#                         fillcolor=f'rgba{(*sns.color_palette("colorblind")[i], 0.3)}',
#                         line=dict(color='rgba(0,0,0,0)'),
#                         hoverinfo="skip",
#                         showlegend=False,
#                         legendgroup=f'{metal}',
#                         xhoverformat=x_formatting,
#                     ))
#         # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
#         fig.update_yaxes(
#             range=[np.amax([y_min, -0.5]), np.amin([y_max, 1.5])]
#         )
#         # Set limits of x-axis to match the edge measurements
#         fig.update_xaxes(
#             range=[np.amin(df[x_axis][df_filter]), 
#             np.amax(df[x_axis][df_filter])]
#             )
#         # Specify the units used on the x-axis
#         if 'Temperature' in x_axis:
#             units = ' [°C]'
#         else:
#             units = ''
#         # Specify text and formatting of axis labels
#         fig.update_layout(
#             xaxis_title=f'<b>{x_axis}{units}</b>',
#             yaxis_title='<b>Weight fraction</b>',
#             font=dict(
#                 size=14,
#             ),
#             hovermode='x unified',
#             legend_title='Experiment',
#         )
#         # Save plot as an image
#         if save_plot:
#             fig.write_image(f'./Data/Plots/{save_name}')
#         fig.show()
#     return None

# #%% Watching directory for changes

# def in_situ_analysis_standards(
#     data_paths: Union[str, list[str]],
#     synchrotron: str,
#     edge_correction_energies: dict,
#     df_foils: pd.DataFrame,
#     df_precursors: pd.DataFrame,
#     metal: str,
#     precursor_suffix: str,
#     x_axis: str='Measurement',
#     file_selection_condition: str='*',
#     negated_condition: bool=False,
#     use_preedge: bool=True,
#     use_transmission: bool=False,
#     interactive: bool=False,
#     with_uncertainty: bool=True,
# ) -> None:
#     """Function that runs the entire analysis pipeline for in-situ analysis with measured standards.

#     Args:
#         data_paths (Union[str, list[str]]): Path or list of paths to folders containing the measured data.
#         synchrotron (str): Name of the synchrotron the data was measured at.
#         edge_correction_energies (dict): Energy shifts for all relevant edges.
#         df_foils (pd.DataFrame): Normalized measurements of the reduced metal edges.
#         df_precursors (pd.DataFrame): Normalized measurements of the unreduced precursors.
#         metal (str): The metal to use for plotting.
#         precursor_suffix (str): The precursor to use for plotting.
#         x_axis (str, optional): The column to use for the x-axis when plotting. Defaults to 'Measurement'.
#         file_selection_condition (str, optional): Pattern to match with filenames. Defaults to '*'.
#         negated_condition (bool, optional): Whether the filenames matching the pattern is included or excluded. Defaults to False.
#         use_preedge (bool, optional): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
#         use_transmission (bool, optional): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
#         interactive (bool, optional): Boolean flag deciding if plots are interactive or static. Defaults to False.
#         with_uncertainty (bool, optional): Boolean flag deciding if the uncertainties are plotted. Defaults to True.

#     Returns:
#         None
#     """
#     # Clears the notebook cell output when this function is called.
#     clear_output(wait=True)
#     # Read the in-situ data
#     if type(data_paths) == list and len(data_paths) > 1:
#         # Create empty list to hold all datasets
#         list_of_datasets = []
#         # Load data
#         for path in data_paths:
#             df_data = load_xas_data(
#                 path, 
#                 synchrotron=synchrotron, 
#                 file_selection_condition=file_selection_condition, 
#                 negated_condition=negated_condition, 
#                 verbose=False,
#             )
#             # Initial data processing
#             df_data = processing_df(df_data, synchrotron=synchrotron)
#             # Append to list of datasets
#             list_of_datasets.append(df_data)
#         # Combine the datasets
#         df_data = combine_datasets(list_of_datasets)
#     else:
#         if type(data_paths) == list:
#             data_paths = data_paths[0]
#         # Load data
#         df_data = load_xas_data(
#             data_paths, 
#             synchrotron=synchrotron, 
#             file_selection_condition=file_selection_condition, 
#             negated_condition=negated_condition, 
#         )
#         # Initial data processing
#         df_data = processing_df(df_data, synchrotron=synchrotron)
#     # Normalization of the data
#     normalize_data(
#         df_data, 
#         edge_correction_energies, 
#         subtract_preedge=use_preedge, 
#         transmission=use_transmission
#     )
#     # Plotting of normalized data
#     plot_data(
#         df_data, 
#         metal=metal, 
#         foils=df_foils, 
#         precursors=df_precursors, 
#         precursor_suffix=precursor_suffix, 
#         interactive=interactive
#     )
#     # Linear combination analysis (LCA)
#     df_results = linear_combination_analysis(df_data, df_foils, df_precursors)
#     # Plot temperature curve
#     plot_temperatures(
#         df_results, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     # Plot LCA over time for single edge
#     plot_LCA_change(
#         df_results, 
#         metal=metal, 
#         precursor_suffix=precursor_suffix, 
#         x_axis=x_axis, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     # Plot LCA over time for all edges
#     plot_reduction_comparison(
#         df_results, 
#         precursor_type='all', 
#         x_axis=x_axis, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     return None

# def in_situ_analysis_averages(
#     data_paths: Union[str, list[str]],
#     synchrotron: str,
#     measurements_to_foil: Union[int, list, np.ndarray, range],
#     measurements_to_precursor: Union[int, list, np.ndarray, range],
#     metal: str,
#     precursor_suffix: str,
#     x_axis: str='Measurement',
#     file_selection_condition: str='*',
#     negated_condition: bool=False,
#     use_preedge: bool=True,
#     use_transmission: bool=False,
#     interactive: bool=False,
#     with_uncertainty: bool=True,
# ) -> None:
#     """Function that runs the entire analysis pipeline for in-situ analysis with measured standards.

#     Args:
#         data_paths (Union[str, list[str]]): Path or list of paths to folders containing the measured data.
#         synchrotron (str): Name of the synchrotron the data was measured at.
#         measurements_to_foil (Union[int, list, np.ndarray, range]): The number of measurements or measurement indeces to average when creating a reference spectra for the reduced metal. 
#         measurements_to_precursor (Union[int, list, np.ndarray, range]): The number of measurements or measurement indeces to average when creating a reference spectra for the unreduced precursor.
#         metal (str): The metal to use for plotting.
#         precursor_suffix (str): The precursor to use for plotting.
#         x_axis (str, optional): The column to use for the x-axis when plotting. Defaults to 'Measurement'.
#         file_selection_condition (str, optional): Pattern to match with filenames. Defaults to '*'.
#         negated_condition (bool, optional): Whether the filenames matching the pattern is included or excluded. Defaults to False.
#         use_preedge (bool, optional): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
#         use_transmission (bool, optional): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
#         interactive (bool, optional): Boolean flag deciding if plots are interactive or static. Defaults to False.
#         with_uncertainty (bool, optional): Boolean flag deciding if the uncertainties are plotted. Defaults to True.

#     Returns:
#         None
#     """
#     # Clears the notebook cell output when this function is called.
#     clear_output(wait=True)
#     # Read the in-situ data
#     if type(data_paths) == list and len(data_paths) > 1:
#         # Create empty list to hold all datasets
#         list_of_datasets = []
#         # Load data
#         for path in data_paths:
#             df_data = load_xas_data(
#                 path, 
#                 synchrotron=synchrotron, 
#                 file_selection_condition=file_selection_condition, 
#                 negated_condition=negated_condition, 
#                 verbose=False,
#             )
#             # Initial data processing
#             df_data = processing_df(df_data, synchrotron=synchrotron, metal=metal)
#             # Append to list of datasets
#             list_of_datasets.append(df_data)
#         # Combine the datasets
#         df_data = combine_datasets(list_of_datasets)
#     else:
#         if type(data_paths) == list:
#             data_paths = data_paths[0]
#         # Load data
#         df_data = load_xas_data(
#             data_paths, 
#             synchrotron=synchrotron, 
#             file_selection_condition=file_selection_condition, 
#             negated_condition=negated_condition,
#         )
#         # Initial data processing
#         df_data = processing_df(df_data, synchrotron=synchrotron, metal=metal)
#     # Convert number of measurements to use into range over the last measurements in the data
#     if type(measurements_to_foil) == int:
#         n_measurements = np.amax(df_data['Measurement'].unique()) + 1
#         measurements_to_foil = range(n_measurements - measurements_to_foil, n_measurements)
#     # Create dataframe with the reference spectra for reduced metals
#     df_foils = average_measurements(df_data, measurements_to_foil)
#     # Calculate the edge energy shift at each edge
#     edge_correction_energies = {
#     'Pd':calc_edge_correction(df_foils, metal='Pd', edge='K', transmission=use_transmission),
#     'Ag':calc_edge_correction(df_foils, metal='Ag', edge='K', transmission=use_transmission),
#     'Rh':calc_edge_correction(df_foils, metal='Rh', edge='K', transmission=use_transmission),
#     'Ru':calc_edge_correction(df_foils, metal='Ru', edge='K', transmission=use_transmission),
#     'Mn':calc_edge_correction(df_foils, metal='Mn', edge='K', transmission=use_transmission),
#     'Mo':calc_edge_correction(df_foils, metal='Mo', edge='K', transmission=use_transmission),
#     'Ir':calc_edge_correction(df_foils, metal='Ir', edge='L3', transmission=use_transmission),
#     'Pt':calc_edge_correction(df_foils, metal='Pt', edge='L3', transmission=use_transmission),
#     }
#     # Normalization of the foil average
#     normalize_data(
#         df_foils, 
#         edge_correction_energies, 
#         subtract_preedge=use_preedge, 
#         transmission=use_transmission,
#     )
#     print(f'Reference spectra from measurement {measurements_to_foil} created.')
#     # Convert number of measurements to use into range over the first measurements in the data
#     if type(measurements_to_precursor) == int:
#         measurements_to_precursor = range(0, measurements_to_precursor)
#     # Create dataframe with the reference spectra for unreduced precursors
#     df_precursors = average_measurements(df_data, measurements_to_precursor)
#     # Normalization of the precursor average
#     normalize_data(
#         df_precursors, 
#         edge_correction_energies, 
#         subtract_preedge=use_preedge, 
#         transmission=use_transmission,
#     )
#     print(f'Reference spectra from measurement {measurements_to_precursor} created.')
#     # Normalization of the data
#     normalize_data(
#         df_data, 
#         edge_correction_energies, 
#         subtract_preedge=use_preedge, 
#         transmission=use_transmission
#     )
#     # Plotting of normalized data
#     plot_data(
#         df_data, 
#         metal=metal, 
#         foils=df_foils, 
#         precursors=df_precursors, 
#         precursor_suffix=precursor_suffix, 
#         interactive=interactive,
#     )
#     # Linear combination analysis (LCA)
#     df_results = linear_combination_analysis(df_data, df_foils, df_precursors)
#     # Plot temperature curve
#     plot_temperatures(
#         df_results, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     # Plot LCA over time for single edge
#     plot_LCA_change(
#         df_results, 
#         metal=metal, 
#         precursor_suffix=precursor_suffix, 
#         x_axis=x_axis, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     # Plot LCA over time for all edges
#     plot_reduction_comparison(
#         df_results, 
#         precursor_type='all', 
#         x_axis=x_axis, 
#         with_uncertainty=with_uncertainty, 
#         interactive=interactive
#     )
#     return None

# class EventHandler(PatternMatchingEventHandler):
#     def __init__(
#         self, 
#         function: Any,
#         function_arguments: dict,
#         patterns: Union[list[str], None]=None, 
#         ignore_patterns: Union[list[str], None]=None, 
#         ignore_directories: bool=False, 
#         case_sensitive: bool=False,
#         sleep_time: Union[int,float]=1,
#         ) -> None:
#         """Class instance that defines how events are handled.

#         Args:
#             function (Any): The function to run when a change happens.
#             function_arguments (dict): The arguments parsed to the function.
#             patterns (Union[list[str], None], optional): Pattern included in the filenames of watched files. Defaults to None.
#             ignore_patterns (Union[list[str], None], optional): Pattern included in the filenames of ignored files. Defaults to None.
#             ignore_directories (bool, optional): Whether changes to directories are ignored. Defaults to False.
#             case_sensitive (bool, optional): Whether the filesystem is case sensitive. Defaults to False.
#             sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.
#         """        
#         # Pass arguments to parent class
#         super().__init__(
#             patterns, 
#             ignore_patterns, 
#             ignore_directories, 
#             case_sensitive
#         )
#         # Assign given arguments
#         self.function = function
#         self.function_arguments = function_arguments
#         self.sleep_time = sleep_time
    
#     # Functions describing the action to take when an event happens in the observed directory
#     # Call a function when a file is modified
#     # def on_modified(self, event) -> None:
#     #     """Executes some action when a file is modified.

#     #     Args:
#     #         event: event instance.

#     #     Returns:
#     #         None
#     #     """        
#     #     self.function(**self.function_arguments)
#     #     time.sleep(self.sleep_time)
#     #     return None
#     # Call a function when a file is created
#     def on_created(self, event) -> None:
#         """Executes some action when a file is created.

#         Args:
#             event: event instance.

#         Returns:
#             None
#         """        
#         self.function(**self.function_arguments)
#         time.sleep(self.sleep_time)
#         return None
#     # Call a function when any event happens
#     # def on_any_event(self, event) -> None:
#     #     """Executes some action when any event happens.

#     #     Args:
#     #         event: event instance.

#     #     Returns:
#     #         None
#     #     """        
#     #     self.function(**self.function_arguments)
#     #     time.sleep(self.sleep_time)
#     #     return None

# def watch_insitu_experiment(
#     data_path: str,
#     function: Any,
#     function_arguments: dict,
#     patterns: Union[list[str], None]=['*'],
#     ignore_patterns: Union[list[str], None]=None,
#     ignore_directories: bool=False,
#     case_sensitive: bool=False,
#     recursive: bool=True,
#     sleep_time: Union[int,float]=1,
# ) -> None:
#     """Continuously watch a directory for changes and re-run analysis code when a relevant change happens.

#     Args:
#         data_path (str): Path to the directory to watch for changes.
#         function (Any): The function to run when a change happens.
#         function_arguments (dict): The arguments parsed to the function.
#         patterns (Union[list[str], None], optional): Pattern included in the filenames of watched files. Defaults to ['*'].
#         ignore_patterns (Union[list[str], None], optional): Pattern included in the filenames of ignored files. Defaults to None.
#         ignore_directories (bool, optional): Whether changes to directories are ignored. Defaults to False.
#         case_sensitive (bool, optional): Whether the filesystem is case sensitive. Defaults to False.
#         recursive (bool, optional): True if events will be emitted for sub-directories traversed recursively; False otherwise. Defaults to True.
#         sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.

#     Returns:
#         None
#     """
#     # Create event handler
#     event_handler = EventHandler(
#         function=function,
#         function_arguments=function_arguments,
#         patterns=patterns, 
#         ignore_patterns=ignore_patterns, 
#         ignore_directories=ignore_directories, 
#         case_sensitive=case_sensitive,
#         sleep_time=sleep_time,
#     )
#     # Create observer
#     observer = Observer()
#     # Tell the observer where to look and what to do
#     observer.schedule(
#         event_handler=event_handler,
#         path=data_path,
#         recursive=recursive
#     )
#     # Start observing
#     observer.start()
#     try:
#         while True:
#             time.sleep(sleep_time)
#     # Ensure proper shutdown of the observer
#     except KeyboardInterrupt:
#         observer.stop()
#         print(f'\nNo longer observing {data_path}')
#     observer.join()
#     return None

# def watch_insitu_experiment_v2(
#     data_path: str,
#     function: Any,
#     function_arguments: dict,
#     pattern: str='*',
#     sleep_time: Union[int,float]=1,
# ) -> None:
#     """Continuously watch a directory for changes and re-run analysis code when a relevant change happens.

#     Args:
#         data_path (str): Path to the directory to watch for changes.
#         function (Any): The function to run when a change happens.
#         function_arguments (dict): The arguments parsed to the function.
#         pattern (str, optional): Pattern included in the filenames of watched files. Defaults to ['*'].
#         sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.

#     Returns:
#         None
#     """
#     # Tell the observer where to look and what to do
#     folder_to_watch = Path(data_path)
#     # Start observing
#     last_change = None
#     try:
#         # Continuously watch for changes
#         while True:
#             # Find the most recently modified file
#             latest_path = max(folder_to_watch.glob(pattern), key=lambda path: path.stat().st_mtime)
#             # Get the time of modification
#             latest_change = latest_path.stat().st_mtime
#             # Check if the modification is new
#             if latest_change != last_change:
#                 # Call function upon change
#                 function(**function_arguments)
#                 # Save the time of modification
#                 last_change = latest_change
#             # Wait before looking for changes again
#             time.sleep(sleep_time)
#     # Ensure proper shutdown of the observer
#     except KeyboardInterrupt:
#         print(f'\nNo longer observing {data_path}')
#     return None