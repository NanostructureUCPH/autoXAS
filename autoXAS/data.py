#%% Imports

# Packages for Regular Expressions
import re
# Packages for handling time
from datetime import datetime
# Packages for math
import numpy as np
# Packages for typing
from typing import Union
# Packages for handling data
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pathlib import Path
from itertools import islice
# from tqdm.auto import tqdm # Use as standard. If progress bar is not rendering use normal tqdm below.
from tqdm import tqdm
#XAS specfific packages
from larch.xray import xray_edge
from larch.xafs import pre_edge, find_e0
from larch import Group

#%% Data loader functions

def load_and_prepare_data(
    folder_path: str,
    energy_column: str,
    I0_columns: Union[str, list],
    I1_columns: Union[str, list],
    metal: Union[str, tuple],
    precursor: Union[None, str, tuple]=None,
    xas_mode: str='Flourescence',
    temperature_column: Union[None, str]=None,
    energy_column_unitConversion: int=1,
    extract_time: bool=False,
    time_regex: str='\S{3}[ ]\S{3}[ ]\d{2}[ ]\d{2}[:]\d{2}[:]\d{2}[ ]\d{4}',
    time_format: str='%a %b %d %H:%M:%S %Y',
    time_startTag: str='',
    time_endTag: str='',
    time_skipLines: int=0,
    file_selection_condition: Union[str,list[str]]='',
    negated_condition: bool=False,
    keep_incomplete: bool=True,
    verbose: bool=True,
) -> pd.DataFrame:
    xas_modes = ['Flourescence', 'F', 'Transmission', 'T']
    assert xas_mode in xas_modes, f'Invalid XAS mode.\nValid modes are {xas_modes}'
    # Compile regex pattern
    compiled_regex = re.compile(time_regex)
    # Find all valid filepaths
    if type(file_selection_condition) == list:
        if negated_condition:
            filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring not in path.stem for substring in file_selection_condition)]
        else:
            filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring in path.stem for substring in file_selection_condition)]
    else:
        if negated_condition:
            filepaths = [path for path in Path(folder_path).glob('*.dat') if not file_selection_condition in path.stem]
        else:
            filepaths = [path for path in Path(folder_path).glob('*.dat') if file_selection_condition in path.stem]
    # Create list to hold data
    list_of_df = []
    # Create list to check if experiment was stopped during a measurement
    list_of_n_measurements = []
    # Define progress bar
    pbar = tqdm(filepaths, desc='Loading data')
    # Loop over all files
    for file in pbar:
        # Used to track which file is being read
        file_name = file.name
        # Used to extract the relevant chemical compound for the measurements
        experiment_name = file.stem
        # Update progress bar
        pbar.set_postfix_str(f'Currently loading {file_name}')
        # Detect size of header and column names
        rows_to_skip = 0
        column_names = []
        # Open the .dat file
        with open(file) as f:
            # Loop through the lines 1-by-1
            for line in f:
                # If the line does not start with a "#" and isn't blank we have reached the data and we end the loop.
                if '#' not in line and len(line) >= 10:
                    break
                # If the line starts with a "#" or is blank we are in the header.
                elif '#' in line or len(line) < 10:
                    # Count the number of rows to skip
                    rows_to_skip += 1
                    # Clean up the line
                    line = line.replace('\n','').split(' ')
                    # Extract the column names
                    column_names = [column for column in line if column][1:]
        # Read the .dat file into a dataframe
        df = pd.read_csv(
            file, 
            sep=' ',
            header=None,
            names=column_names,
            skiprows=rows_to_skip,
            skip_blank_lines=True,
            on_bad_lines='skip',
            keep_default_na=False,
            )
        # Convert the column values to floats
        df[column_names] = df[column_names].apply(pd.to_numeric, errors='coerce', downcast='float')
        # Remove any rows that contained non-numeric data
        df.dropna(axis=0, inplace=True)
        # Log the filename
        df['Filename'] = file_name
        # Log the experiment name
        df['Experiment'] = experiment_name
        # Log the measured metal
        if isinstance(metal, str):
            df['Metal'] = metal
        elif isinstance(metal, tuple):
            df['Metal'] = experiment_name.split(metal[0])[metal[1]]
        # Log the measured precursor
        if isinstance(precursor, str):
            df['Precursor'] = precursor
        elif isinstance(precursor, tuple):
            df['Precursor'] = experiment_name.split(precursor[0])[precursor[1]]
        else:
            df['Precursor'] = None
        # Convert energy to eV
        df['Energy'] = df[energy_column] * energy_column_unitConversion
        # Get temperature
        if temperature_column:
            df['Temperature'] = df[temperature_column]
        else:
            df['Temperature'] = 0
        # Calculate I0
        if isinstance(I0_columns, list):
            df['I0'] = 0
            for column in I0_columns:
                df['I0'] += df[column]
        elif isinstance(I0_columns, str):
            df['I0'] = df[I0_columns]
        # Calculate I1
        if isinstance(I1_columns, list):
            df['I1'] = 0
            for column in I1_columns:
                df['I1'] += df[column]
        elif isinstance(I1_columns, str):
            df['I1'] = df[I1_columns]
        # Calculate absorption coefficient
        if xas_mode in ['Flourescence', 'F']:
            df['Flourescence'] = df['I1'] / df['I0']
            df['Transmission'] = 0
        elif xas_mode in ['Transmission', 'T']:
            df['Transmission'] = np.log( df['I0'] / df['I1'] )
            df['Flourescence'] = 0
        # Determine measurement numbers
        # Calculate time differences in column containing relative time measurements
        difference = df[energy_column].round(2).diff()
        # Create list to hold the measurement numbers
        measurement = []
        # The current measurement number. We start at 1
        measurement_number = 1
        # The first datapoint has no defined difference, so we add it to measurement 1 now
        measurement.append(measurement_number)
        # Loop over the differences in time since measurement started
        for diff_val in difference:
            # If the difference is negative we have started a new measurement
            if diff_val < 0:
                # Increase the current measurement number
                measurement_number += 1
                measurement.append(measurement_number)
            # If the difference is positive we are in the same measurement
            elif diff_val >= 0:
                measurement.append(measurement_number)
        df['Measurement'] = measurement
        # Extract timestamps from file
        if extract_time:
            with open(file) as f:
                lines = f.readlines()
                start_times = [
                    datetime.strptime(compiled_regex.findall(line)[0], time_format) 
                    for i, line in enumerate(lines) 
                    if (compiled_regex.findall(line) != []) and (i >= time_skipLines) and (time_startTag in line)
                ]
                end_times = [
                    datetime.strptime(compiled_regex.findall(line)[0], time_format) 
                    for i, line in enumerate(lines) 
                    if (compiled_regex.findall(line) != []) and (i >= time_skipLines) and (time_endTag in line)
                ]
            # Save start and end times in dataframe
            df['Start Time'] = start_times
            df['End Time'] = end_times
            # Calculate measurement duration
            df['Relative Time'] = df['End Time'] - df['Start Time']
        else:
            # Save start and end times in dataframe
            df['Start Time'] = 0
            df['End Time'] = 1
            # Calculate measurement duration
            df['Relative Time'] = df['End Time'] - df['Start Time']
        # Append dataframe to the list
        list_of_df.append(df)
        # Log the number of measurements
        list_of_n_measurements.append(measurement_number)
    # Merge all the dataframes into one dataset
    df = pd.concat(list_of_df)
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    # Insert empty columns for normalization
    df['Energy_Corrected'] = 0
    df['Normalized'] = 0
    df['pre_edge'] = 0
    df['post_edge'] = 0
    # Log the number of measurements in each experiment
    # Remove incomplete measurements
    if np.amin(list_of_n_measurements) != np.amax(list_of_n_measurements):
        if verbose:
            print('Incomplete measurement detected!')
            print(f'Not all edges were measured {np.amax(list_of_n_measurements)} times, but only {np.amin(list_of_n_measurements)} times.')
            print('Incomplete measurements will be removed unless keep_incomplete="True".')
        if not keep_incomplete:
            df.drop(df[df['Measurement'] > np.amin(list_of_n_measurements)].index, inplace=True)
            if verbose:
                print('\nIncomplete measurements were removed!')
    return df

def load_xas_data(
    folder_path: str, 
    synchrotron: str, 
    keep_incomplete: bool=False,
    file_selection_condition: Union[str,list[str]]='',
    negated_condition: bool=False,
    verbose: bool=True,
) -> pd.DataFrame:
    """------------------------------------------------------------------
    Load XAS data files (.dat extension) from a folder.

    Args:
        folder_path (str): Filepath to the folder containing the .dat files.
        synchrotron (str): Name of the synchrotron the data was measured at.
        keep_incomplete (optional, bool): Whether to keep incomplete measurements or not. Defaults to False.
        file_selection_condition (optional, Union[str, list[str]]): Substring present in filenames to either load or ignore. Defaults to ' '.
        negated_condition (optional, bool): Whether to load or ignore files with the given substring in the filename. Defaults to False.
        verbose (optional, bool): Whether to print other things than progress bars. Defaults to True.

    Returns:
        pd.DataFrame: The raw data from the .dat files.
    """    
    # List of the synchrotrons which the function can load data from
    implemented = ['ESRF', 'BALDER', 'BALDER_2', 'SNBL']
    # Check the data format can be handled
    assert synchrotron in implemented, f'Loading of data from {synchrotron} is not implemented. The implemented synchrotrons are:\n\t{implemented}\n\nIf you want the loading of data from a specific facility implemented contact me at ufj@chem.ku.dk or submit a request at [github link].'
    if synchrotron in ['ESRF', 'BALDER', 'BALDER_2', 'SNBL']:
        # Reads all the correct files
        if type(file_selection_condition) == list:
            if negated_condition:
                filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring not in path.stem for substring in file_selection_condition)]
            else:
                filepaths = [path for path in Path(folder_path).glob('*.dat') if all(substring in path.stem for substring in file_selection_condition)]
        else:
            if negated_condition:
                filepaths = [path for path in Path(folder_path).glob('*.dat') if not file_selection_condition in path.stem]
            else:
                filepaths = [path for path in Path(folder_path).glob('*.dat') if file_selection_condition in path.stem]
        # Create list to hold data
        list_of_df = []
        # Create list to check if experiment was stopped during a measurement
        list_of_n_measurements = []
        # Define progress bar
        pbar = tqdm(filepaths, desc='Loading data')
        # Loop over all files
        for file in pbar:
            # Used to track which file is being read
            file_name = file.name
            # Used to extract the relevant chemical compound for the measurements
            experiment_name = file.stem
            # Update progress bar
            pbar.set_postfix_str(f'Currently loading {file_name}')
            # Detect size of header and column names
            rows_to_skip = 0
            column_names = []
            # Open the .dat file
            with open(file) as f:
                # Loop through the lines 1-by-1
                for line in f:
                    # If the line does not start with a "#" and isn't blank we have reached the data and we end the loop.
                    if '#' not in line and len(line) >= 10:
                        break
                    # If the line starts with a "#" or is blank we are in the header.
                    elif '#' in line or len(line) < 10:
                        # Count the number of rows to skip
                        rows_to_skip += 1
                        # Clean up the line
                        line = line.replace('\n','').split(' ')
                        # Extract the column names
                        column_names = [column for column in line if column][1:]
            # Read the .dat file into a dataframe
            df = pd.read_csv(
                file, 
                sep=' ',
                header=None,
                names=column_names,
                skiprows=rows_to_skip,
                skip_blank_lines=True,
                on_bad_lines='skip',
                keep_default_na=False,
                )
            # Convert the column values to floats
            df[column_names] = df[column_names].apply(pd.to_numeric, errors='coerce', downcast='float')
            # Remove any rows that contained non-numeric data
            df.dropna(axis=0, inplace=True)
            # Log the filename
            df['Filename'] = file_name
            # Do beamline/synchrotron specific things
            if synchrotron in ['BALDER']:
                # The current measurement number.
                measurement = np.int(experiment_name.split('_')[-1])
                # Add the measurement and experiment information to the dataframe
                df['Measurement'] = measurement
                df['Experiment'] = '_'.join(experiment_name.split('_')[:-1])
                # Append dataframe to the list
                list_of_df.append(df)
                # Tags defining lines with timestamps
                start_tag = '#C Acquisition started'
                end_tag = '#C Acquisition ended'
                # Timestamp format
                time_format = '%a %b %d %H:%M:%S %Y'
                # Extract timestamps
                with open(file) as f:
                    # Read all lines
                    lines = f.readlines()
                    # Find line with starting time
                    start_time = [line[26:].replace('\n', '') for line in lines if start_tag in line][0]
                    # Format the line into a datetime object
                    start_time = datetime.strptime(start_time, time_format)
                    # Save the start time
                    df['Start Time'] = start_time
                    # Find the line with ending time
                    end_time = [line[24:].replace('\n', '') for line in lines if end_tag in line][0]
                    # Format the line into a datetime object
                    end_time = datetime.strptime(end_time, time_format)
                    # Save the end time
                    df['End Time'] = end_time
            elif synchrotron in ['ESRF', 'BALDER_2', 'SNBL']:
                # Detect what data belongs to different measurements
                # Find the difference in time since measurement started
                if synchrotron in ['ESRF']:
                    difference = df['Htime'].diff()
                elif synchrotron in ['SNBL']:
                    difference = df['ZapEnergy'].round(2).diff()
                elif synchrotron in ['BALDER_2']:
                    difference = df['dt'].diff()
                    # Create lists to hold times
                    list_start_times = []
                    list_end_times = []
                    # Tags defining lines with timestamps
                    start_tag = '#C Acquisition started'
                    end_tag = '#C Acquisition ended'
                    # Timestamp format
                    time_format = '%a %b %d %H:%M:%S %Y'
                    # Extract timestamps
                    with open(file) as f:
                        # Read all lines
                        lines = f.readlines()
                        # Find line with starting time
                        start_times = [line[26:].replace('\n', '') for line in lines if start_tag in line]
                        # Format the line into a datetime object
                        start_times = [datetime.strptime(time, time_format) for time in start_times]
                        # Find the line with ending time
                        end_times = [line[24:].replace('\n', '') for line in lines if end_tag in line]
                        # Format the line into a datetime object
                        end_times = [datetime.strptime(time, time_format) for time in end_times]
                # Create list to hold the measurement numbers
                measurement = []
                # The current measurement number. We start at 1
                measurement_number = 1
                # The first datapoint has no defined difference, so we add it to measurement 1 now
                measurement.append(measurement_number)
                # Loop over the differences in time since measurement started
                for diff_val in difference:
                    # If the difference is negative we have started a new measurement
                    if diff_val < 0:
                        # Increase the current measurement number
                        measurement_number += 1
                        measurement.append(measurement_number)
                    # If the difference is positive we are in the same measurement
                    elif diff_val >= 0:
                        measurement.append(measurement_number)
                    if synchrotron in ['BALDER_2']:
                        # Append the times
                        list_start_times.append(start_times[measurement_number - 1])
                        list_end_times.append(end_times[measurement_number - 1])
                # Add the measurement and experiment information to the dataframe
                df['Measurement'] = measurement
                if '_ref' in experiment_name:
                    df['Experiment'] = '_'.join(experiment_name.split('_')[:-1])
                else:
                    df['Experiment'] = experiment_name
                # Add times to the dataframe
                if synchrotron in ['BALDER_2']:
                    df['Start Time'] = list_start_times
                    df['End Time'] = list_end_times
                # Append dataframe to the list
                list_of_df.append(df)
                # Log the number of measurements
                list_of_n_measurements.append(measurement_number)
        # Merge all the dataframes into one dataset
        df = pd.concat(list_of_df)
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        # Log the number of measurements in each experiment
        if synchrotron in ['BALDER']:
            # Loop over the experiments
            for experiment in df['Experiment'].unique():
                # Find the number of measurements in the experiment
                n_measurements = np.amax(df['Measurement'][df['Experiment'] == experiment])
                # Log the number of measurements in the experiment
                list_of_n_measurements.append(n_measurements)
        # Remove incomplete measurements
        if np.amin(list_of_n_measurements) != np.amax(list_of_n_measurements):
            if verbose:
                print('Incomplete measurement detected!')
                print(f'Not all edges were measured {np.amax(list_of_n_measurements)} times, but only {np.amin(list_of_n_measurements)} times.')
                print('Incomplete measurements will be removed unless keep_incomplete="True".')
            if not keep_incomplete:
                df.drop(df[df['Measurement'] > np.amin(list_of_n_measurements)].index, inplace=True)
                if verbose:
                    print('\nIncomplete measurements were removed!')
    return df
#%% Saving data

def save_data(
    data: pd.DataFrame,
    filename: str='XAS_data.csv',
    save_folder: str='./Data/SavedData/',
) -> None:
    """Function to save dataframe to .csv file for use in other analysis or plotting programs.

    Args:
        data (pd.DataFrame): Dataframe to save.
        filename (optional, str): Filename to use for the saved file. Defaults to 'XAS_data.csv'.
        save_folder (optional, str): Folder to save the file in. Defaults to './Data/SavedData/'.

    Returns:
        None
    """    
    # Create the save folder if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    # Save the file
    data.to_csv(save_folder + filename, index=False)
    return None

#%% Preprocessing functions

def processing_df(
    df: pd.DataFrame,
    synchrotron: str,
    metal: Union[str, None]=None,
    precursor: Union[str, None]=None,
    high_energy_metals: list=['Pd'],
) -> pd.DataFrame:
    """------------------------------------------------------------------
    Initial preprocessing of XAS data.

    Select relevant data columns, calculate absorption and transmission signals, and initialize empty columns used for normalization.

    Args:
        df (pd.DataFrame): The raw input data.
        synchrotron (str): Name of the synchrotron the data was measured at.
        metal (optional, Union[str, None]): The metal that was measured. Defaults to None.
        precursor (optional, Union[str, None]): The precursor that was measured. Defaults to None.

    Returns:
        pd.DataFrame: The cleaned and preprocessed data.
    """    
     # List of the synchrotrons which the function can load data from
    implemented = ['ESRF', 'BALDER', 'BALDER_2', 'SNBL']
    # Check the data format can be handled
    assert synchrotron in implemented, f'Loading of data from {synchrotron} is not implemented. The implemented synchrotrons are:\n\t{implemented}\n\nIf you want the loading of data from a specific facility implemented contact me at ufj@chem.ku.dk or submit a request at [github link].'
    if synchrotron in ['ESRF']:
        # Select the relevant columns
        df_new = df[['Filename', 'Experiment', 'Measurement', 'ZapEnergy', 'MonEx', 'xmap_roi00', 'Ion1']]
        # Assign the measured metal or infer from file
        if metal != None:
            df_new['Metal'] = metal
        else:
            df_new['Metal'] = df['Experiment'].str[:2]
        # Assign the measured precursor counter-ion or infer from file
        if precursor != None:
            df_new['Precursor'] = precursor
        elif len(df['Experiment'][0]) > 2:
            df_new['Precursor'] = df['Experiment'].str[2:]
        else:
            df_new['Precursor'] = precursor
        # Calculate the correct x- and y-values for looking at the measured Flourescence and transmission data
        df_new['Energy'] = df_new['ZapEnergy'] * 1000
        df_new['Temperature'] = df['Nanodac']
        df_new['Flourescence'] = df_new['xmap_roi00'].to_numpy() / df_new['MonEx'].to_numpy()
        df_new['Transmission'] = np.log(df_new['MonEx'].to_numpy() / df_new['Ion1'].to_numpy())
        df_new['Relative Time'] = 0
    elif synchrotron in ['BALDER']:
        # Select the relevant columns
        df_new = df[['Filename', 'Experiment', 'Measurement', 'Start Time', 'End Time', 'albaem01_ch1', 'albaem01_ch2', 'albaem02_ch3', 'albaem02_ch4']]
        # Assign the measured metal or infer from file
        if metal != None:
            df_new['Metal'] = metal
            # Re-assign the experiment name
            df_new['Experiment'] = metal
        else:
            raise ValueError('The measured metal can not be inferred from the data format at BALDER')
        # Assign the measured precursor counter-ion. It can not be inferred from file
        df_new['Precursor'] = precursor
        if precursor != None:
            # Re-assign the experiment name
            df_new['Experiment'] += df_new['Precursor']
        # Calculate the correct x- and y-values for looking at the measured Flourescence and transmission data
        df_new['Relative Time'] = (df_new['Start Time'] - df_new['Start Time'][0]).dt.total_seconds()
        df_new['Energy'] = df['mono1_energy']
        df_new['Temperature'] = 0
        df_new['Flourescence'] = 0
        df_new['Transmission'] = ( df['albaem01_ch1'] + df['albaem01_ch2'] ) / ( df['albaem02_ch3'] + df['albaem02_ch4'] )
    elif synchrotron in ['SNBL']:
        # Select the relevant columns
        df_new = df[['Filename', 'Experiment', 'Measurement', 'ZapEnergy', 'xmap_roi00', 'mon_3', 'mon_4']]
        # Assign the measured metal or infer from file
        if metal != None:
            df_new['Metal'] = metal
        else:
            df_new['Metal'] = df['Experiment'].str.split('_').str[-2]
        # Assign the measured precursor counter-ion or infer from file
        if precursor != None:
            df_new['Precursor'] = precursor
        else:
            df_new['Precursor'] = precursor
        # Calculate the correct x- and y-values for looking at the measured Flourescence and transmission data
        df_new['Energy'] = df_new['ZapEnergy'] * 1000
        df_new['Temperature'] = 0
        if metal in high_energy_metals:
            df_new['Flourescence'] = df_new['xmap_roi00'].to_numpy() / df_new['mon_3'].to_numpy()
        else:
            df_new['Flourescence'] = df_new['xmap_roi00'].to_numpy() / df_new['mon_4'].to_numpy()
        df_new['Transmission'] = 0
        df_new['Relative Time'] = 0
    elif synchrotron in ['BALDER']:
        # Select the relevant columns
        df_new = df[['Filename', 'Experiment', 'Measurement', 'Start Time', 'End Time', 'albaem01_ch1', 'albaem01_ch2', 'albaem02_ch3', 'albaem02_ch4']]
        # Assign the measured metal or infer from file
        if metal != None:
            df_new['Metal'] = metal
            # Re-assign the experiment name
            df_new['Experiment'] = metal
        else:
            raise ValueError('The measured metal can not be inferred from the data format at BALDER')
        # Assign the measured precursor counter-ion. It can not be inferred from file
        df_new['Precursor'] = precursor
        if precursor != None:
            # Re-assign the experiment name
            df_new['Experiment'] += df_new['Precursor']
        # Calculate the correct x- and y-values for looking at the measured Flourescence and transmission data
        df_new['Relative Time'] = (df_new['Start Time'] - df_new['Start Time'][0]).dt.total_seconds()
        df_new['Energy'] = df['mono1_energy']
        df_new['Temperature'] = 0
        df_new['Flourescence'] = 0
        df_new['Transmission'] = ( df['albaem01_ch1'] + df['albaem01_ch2'] ) / ( df['albaem02_ch3'] + df['albaem02_ch4'] )
    elif synchrotron in ['BALDER_2']:
        # Select the relevant columns
        df_new = df[['Filename', 'Experiment', 'Measurement', 'Start Time', 'End Time', 'albaem02_ch1', 'albaem02_ch2', 'albaem02_ch3', 'albaem02_ch4']]
        # Assign the measured metal or infer from file
        if metal != None:
            df_new['Metal'] = metal
        else:
            df_new['Metal'] = df['Experiment'].str[:2]
        # Assign the measured precursor counter-ion. It can not be inferred from file
        if precursor != None:
            df_new['Precursor'] = precursor
        elif len(df['Experiment'][0]) > 2:
            df_new['Precursor'] = df['Experiment'].str[2:]
        else:
            df_new['Precursor'] = precursor
        # Calculate the correct x- and y-values for looking at the measured Flourescence and transmission data
        df_new['Relative Time'] = (df_new['Start Time'] - df_new['Start Time'][0]).dt.total_seconds()
        df_new['Energy'] = df['mono1_energy']
        df_new['Temperature'] = 0
        df_new['Flourescence'] = 0
        df_new['Transmission'] = ( df['albaem02_ch1'] + df['albaem02_ch2'] ) / ( df['albaem02_ch3'] + df['albaem02_ch4'] )

    df_new['Energy_Corrected'] = 0
    df_new['Normalized'] = 0
    df_new['pre_edge'] = 0
    df_new['post_edge'] = 0
    return df_new

def calc_edge_correction(
    df: pd.DataFrame, 
    metal: str, 
    edge: str, 
    transmission: bool=False
) -> float:
    """------------------------------------------------------------------
    Calculation of the energy shift between the measured edge and the table value.

    Args:
        df (pd.DataFrame): The preprocessed data.
        metal (str): The measured metal.
        edge (str): The relevant edge (K, L1, L2, L3).
        transmission (optional, bool): Boolean flag deciding if Flourescence (False) or transmission (True) signal is used. Defaults to False.

    Returns:
        float: Edge energy shift
    """    
    # Choose what type of XAS data to work with
    if transmission:
        data_type = 'Transmission'
    else:
        data_type = 'Flourescence'
    # Find the table value for the relevant edge
    edge_table = xray_edge(metal, edge, energy_only=True)
    # Create dataframe filter
    df_filter = (df['Experiment'].str.contains(metal)) & (df['Measurement'] == 1)
    try:
        # Find the measured edge energy
        edge_measured = find_e0(df['Energy'][df_filter].to_numpy(), df[data_type][df_filter].to_numpy())
        # Calculate the needed correction
        correction_value = edge_table - edge_measured
    except:
        correction_value = None
    return correction_value

def normalize_data(
    df: pd.DataFrame, 
    edge_correction_energies: dict, 
    subtract_preedge: bool=True, 
    transmission: bool=False,
) -> None:
    """------------------------------------------------------------------
    Normalization of XAS data.

    The subtraction of pre-edge fit from the data can ruin the normalization, so set subtract_preedge=False if the normalized data looks weird.

    Args:
        df (pd.DataFrame): The preprocessed data.
        edge_correction_energies (dict): Energy shifts for all relevant edges.
        subtract_preedge (optional, bool): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
        transmission (optional, bool): Boolean flag deciding if Flourescence (False) or transmission (True) signal is used. Defaults to False.

    Returns:
        None
    """    
    # Choose what type of XAS data to work with
    if transmission:
        data_type = 'Transmission'
    else:
        data_type = 'Flourescence'
    # Iterate over each experiment
    for experiment in tqdm(df['Experiment'].unique(), desc='Normalization progress: '):
        # Select only relevant values
        exp_filter = (df['Experiment'] == experiment)
        print(experiment)
        # Measured metal
        metal = df['Metal'][exp_filter].to_list()[0]
        print(metal)
        # Correct for the energy shift at the edge
        df['Energy_Corrected'][exp_filter] = df['Energy'][exp_filter] + edge_correction_energies[metal]
        # Iterate over each measurement
        measurement_correction = 0
        for measurement_id in tqdm(df['Measurement'][exp_filter].unique(), leave=False, desc=f'Normalizing {metal}: '):
            measurement_id -= measurement_correction
            # Select only relevant values
            df_filter = (df['Experiment'] == experiment) & (df['Measurement'] == measurement_id)
            # Create dummy Group to collect post edge fit
            g = Group(name='temp_group')
            # Subtract minimum value from measurement
            df['Normalized'][df_filter] = df[data_type][df_filter] - np.amin(df[data_type][df_filter])
            if all(x == 0.0 for x in df['Normalized'][df_filter]):
                raise Exception('')
            try:
                # Fit to the pre- and post-edge
                pre_edge(df['Energy_Corrected'][df_filter].to_numpy(), df['Normalized'][df_filter].to_numpy(), group=g)
                # Save pre- and post-edge for plotting
                df['pre_edge'][df_filter] = g.pre_edge
                df['post_edge'][df_filter] = g.post_edge
                # Subtract pre-edge fit if specified
                if subtract_preedge:
                    df['Normalized'][df_filter] -= g.pre_edge
                    # Re-fit to the post-edge 
                    pre_edge(df['Energy_Corrected'][df_filter].to_numpy(), df['Normalized'][df_filter].to_numpy(), group=g)
                # Normalize with the post edge fit
                df['Normalized'][df_filter] /= g.post_edge
            except:
                print(f'Error occurred during normalization of data from measurement {measurement_id}. The error is most likely due to the measurement being stopped before completion.\nThe measurement (incl. all edges) has therefore been removed from the dataset and measurement numbers are corrected.')
                df.drop(df[(df['Measurement'] == measurement_id)].index, inplace=True)
                df['Measurement'][(df['Measurement'] > measurement_id)] -= 1
                measurement_correction += 1

    return None

def combine_datasets(
    datasets: list[pd.DataFrame]
) -> pd.DataFrame:
    """Combine two or more datasets from the same experiment

    Args:
        datasets (list[pd.DataFrame]): List of the datasets in the correct order

    Returns:
        pd.DataFrame: The combined dataset
    """    
    # Create empty list to hold dataframes
    df_list = []
    # Counter to update measurement values
    measurement_counter = 0
    # Loop over all the datasets
    for df in datasets:
        # Update the measurement number
        df['Measurement'] += measurement_counter
        # Update the measurement counter
        measurement_counter = np.amax(df['Measurement'])
        # Add dataframe to the list
        df_list.append(df)
    # Combine the dataframes
    combined_df = pd.concat(df_list)
    combined_df.reset_index(inplace=True)
    return combined_df

def average_measurements(
    data: pd.DataFrame, 
    measurements_to_average: Union[list[int], np.ndarray, range], 
) -> pd.DataFrame:
    """Calculates the average XAS data for the specified measurements.

    Args:
        data (pd.DataFrame): Dataframe containing the XAS data.
        measurements_to_average (Union[list[int], np.ndarray, range]): The measurements to be averaged.
        repeating (optional, bool): If True every measurements_to_average frames are averaged together. Defaults to False.

    Returns:
        averaged_data (pd.DataFrame): Dataframe containing the averaged XAS data.
    """
    # Create list to hold dataframes for each experiment
    list_of_df = []
    # Loop over all experiments
    for experiment in data['Experiment'].unique():
        # Loop over all the frames to average
        for measurement in measurements_to_average:
            # Select only relevant values
            data_filter = (data['Experiment'] == experiment) & (data['Measurement'] == measurement)
            # Create array to store average
            if measurement == measurements_to_average[0]:
                df_avg = data[data_filter].copy()
                avg_flourescence = np.zeros_like(data['Flourescence'][data_filter], dtype=np.float64)
                avg_transmission = np.zeros_like(data['Transmission'][data_filter], dtype=np.float64)
            # Sum the measurements together
            avg_flourescence += data['Flourescence'][data_filter].to_numpy()
            avg_transmission += data['Transmission'][data_filter].to_numpy()
        # Divide by the number of measurements used
        n_measurements = float(len(measurements_to_average))
        avg_flourescence /= n_measurements
        avg_transmission /= n_measurements
        # Put the averaged data into the dataframe
        df_avg['Flourescence'] = avg_flourescence
        df_avg['Transmission'] = avg_transmission
        # Select only relevant values
        temp_filter = (data['Experiment'] == experiment) & (data['Measurement'].isin(measurements_to_average))
        # Calculate the average temperature for the averaged data
        df_avg['Temperature'] = data['Temperature'][temp_filter].mean()
        # Set the measurement number to 1
        df_avg['Measurement'] = 1
        # Add the dataframe with the averaged data to list
        list_of_df.append(df_avg)
    # Combine all experiments into one dataframe
    df_avg = pd.concat(list_of_df)
    return df_avg

def average_measurements_periodic(
    data: pd.DataFrame,  
    period: Union[None, int]=None,
    n_periods: Union[None, int]=None,
) -> pd.DataFrame:
    """Calculates the average XAS data for the specified measurements.

    Args:
        data (pd.DataFrame): Dataframe containing the XAS data.
        measurements_to_average (Union[list[int], np.ndarray, range]): The measurements to be averaged.
        repeating (optional, bool): If True every measurements_to_average frames are averaged together. Defaults to False.

    Returns:
        averaged_data (pd.DataFrame): Dataframe containing the averaged XAS data.
    """
    # Create list to hold dataframes for each experiment
    list_of_df = []
    # Loop over all experiments
    if (period and n_periods) or (not period and not n_periods) :
        n_arguments = bool(period) + bool(n_periods)
        raise Exception(f"Exactly 1 optional argument should be given. {n_arguments} was given.")
    # Perform the periodic averaging of the data
    for experiment in data['Experiment'].unique():
        if period:
            # Find number of measurements to average
            n_total_measurements = np.amax(data['Measurement'][data['Experiment'] == experiment])
            n_measurements_to_average = period
            new_n_measurements = int(np.ceil(n_total_measurements / period))
        elif n_periods:
            # Find number of measurements to average
            n_total_measurements = np.amax(data['Measurement'][data['Experiment'] == experiment])
            n_measurements_to_average = n_total_measurements // n_periods
            new_n_measurements = n_periods
        measurements_to_average = np.arange(n_measurements_to_average)+1
        measurements_to_average_temp = measurements_to_average.copy()
        for measurement_number in range(new_n_measurements):
            if measurements_to_average_temp.any() >= n_total_measurements:
                measurements_to_average_temp = np.array([i for i in measurements_to_average_temp if i < n_total_measurements])
            # Loop over all the frames to average
            for measurement in measurements_to_average_temp:
                # Select only relevant values
                data_filter = (data['Experiment'] == experiment) & (data['Measurement'] == measurement)
                # Create array to store average
                if measurement == measurements_to_average_temp[0]:
                    df_avg = data[data_filter].copy()
                    avg_flourescence = np.zeros_like(data['Flourescence'][data_filter], dtype=np.float64)
                    avg_transmission = np.zeros_like(data['Transmission'][data_filter], dtype=np.float64)
                # Sum the measurements together
                avg_flourescence += data['Flourescence'][data_filter].to_numpy()
                avg_transmission += data['Transmission'][data_filter].to_numpy()
            # Divide by the number of measurements used
            n_measurements = float(len(measurements_to_average_temp))
            avg_flourescence /= n_measurements
            avg_transmission /= n_measurements
            # Put the averaged data into the dataframe
            df_avg['Flourescence'] = avg_flourescence
            df_avg['Transmission'] = avg_transmission
            # Select only relevant values
            temp_filter = (data['Experiment'] == experiment) & (data['Measurement'].isin(measurements_to_average_temp))
            # Calculate the average temperature for the averaged data
            df_avg['Temperature'] = data['Temperature'][temp_filter].mean()
            # Set the measurement number to 1
            df_avg['Measurement'] = measurement_number + 1
            # Add the dataframe with the averaged data to list
            list_of_df.append(df_avg)
            measurements_to_average_temp += n_measurements_to_average
    # Combine all experiments into one dataframe
    df_avg = pd.concat(list_of_df)
    return df_avg

#%% File splitting

def split_dat_file(
    data_folder: str,
    filename: str,
    header_length: int,
    data_length: int,
    footer_length: int,
    file_extension: str='.dat',
    save_folder: Union[str, None]=None,
) -> None:
    """Function that splits a data file containing multiple measurements into individual files.

    Args:
        data_folder (str): The path to the folder containing the file to split.
        filename (str): The name of the file to split.
        header_length (int): The number of lines in the header.
        data_length (int): The number of lines containing data.
        footer_length (int): The number of lines in the footer.
        file_extension (str, optional): The type of file to split. It should work for common non-binary formats like txt, csv, dat, etc. Defaults to '.dat'.
        save_folder (Union[str, None], optional): The path to the folder to save the new files in. If "None" a subfolder will be created named after the split file. Defaults to None.

    Returns:
        None
    """
    file = data_folder + filename + file_extension
    if save_folder == None:
        # Where to save the data
        save_folder = data_folder + filename + '_split/'
    # Create save folder
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    # Define the number of lines to save in each file
    file_length = header_length + data_length + footer_length
    # Split the file and save as new files
    with open(file) as f:
        for i, measurement in enumerate(iter(lambda:list(islice(f, file_length)), []), 1):
            with open(f'{save_folder}{filename}_{i}{file_extension}', 'w') as split:
                split.writelines(measurement)
    return None