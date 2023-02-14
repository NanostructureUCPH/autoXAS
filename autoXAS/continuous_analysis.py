#%% Imports

# Packages for handling time
import time
# Packages for math
import numpy as np
# Packages for typing
from typing import Any, Union
# Packages for handling data
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pathlib import Path
# Packages for watching directories
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
# Packages for managing notebook and IPython features
from IPython.display import clear_output
# Importing functions from the other scripts
from autoXAS.data import *
from autoXAS.analysis import *
from autoXAS.plotting import *

#%% Watching directory for changes

def in_situ_analysis_standards(
    data_paths: Union[str, list[str]],
    synchrotron: str,
    edge_correction_energies: dict,
    df_foils: pd.DataFrame,
    df_precursors: pd.DataFrame,
    metal: str,
    precursor_suffix: str,
    x_axis: str='Measurement',
    file_selection_condition: str='*',
    negated_condition: bool=False,
    use_preedge: bool=True,
    use_transmission: bool=False,
    interactive: bool=False,
    with_uncertainty: bool=True,
) -> None:
    """Function that runs the entire analysis pipeline for in-situ analysis with measured standards.

    Args:
        data_paths (Union[str, list[str]]): Path or list of paths to folders containing the measured data.
        synchrotron (str): Name of the synchrotron the data was measured at.
        edge_correction_energies (dict): Energy shifts for all relevant edges.
        df_foils (pd.DataFrame): Normalized measurements of the reduced metal edges.
        df_precursors (pd.DataFrame): Normalized measurements of the unreduced precursors.
        metal (str): The metal to use for plotting.
        precursor_suffix (str): The precursor to use for plotting.
        x_axis (str, optional): The column to use for the x-axis when plotting. Defaults to 'Measurement'.
        file_selection_condition (str, optional): Pattern to match with filenames. Defaults to '*'.
        negated_condition (bool, optional): Whether the filenames matching the pattern is included or excluded. Defaults to False.
        use_preedge (bool, optional): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
        use_transmission (bool, optional): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
        interactive (bool, optional): Boolean flag deciding if plots are interactive or static. Defaults to False.
        with_uncertainty (bool, optional): Boolean flag deciding if the uncertainties are plotted. Defaults to True.

    Returns:
        None
    """
    # Clears the notebook cell output when this function is called.
    clear_output(wait=True)
    # Read the in-situ data
    if type(data_paths) == list and len(data_paths) > 1:
        # Create empty list to hold all datasets
        list_of_datasets = []
        # Load data
        for path in data_paths:
            df_data = load_xas_data(
                path, 
                synchrotron=synchrotron, 
                file_selection_condition=file_selection_condition, 
                negated_condition=negated_condition, 
                verbose=False,
            )
            # Initial data processing
            df_data = processing_df(df_data, synchrotron=synchrotron)
            # Append to list of datasets
            list_of_datasets.append(df_data)
        # Combine the datasets
        df_data = combine_datasets(list_of_datasets)
    else:
        if type(data_paths) == list:
            data_paths = data_paths[0]
        # Load data
        df_data = load_xas_data(
            data_paths, 
            synchrotron=synchrotron, 
            file_selection_condition=file_selection_condition, 
            negated_condition=negated_condition, 
        )
        # Initial data processing
        df_data = processing_df(df_data, synchrotron=synchrotron)
    # Normalization of the data
    normalize_data(
        df_data, 
        edge_correction_energies, 
        subtract_preedge=use_preedge, 
        transmission=use_transmission
    )
    # Plotting of normalized data
    plot_data(
        df_data, 
        metal=metal, 
        foils=df_foils, 
        precursors=df_precursors, 
        precursor_suffix=precursor_suffix, 
        interactive=interactive
    )
    # Linear combination analysis (LCA)
    df_results = linear_combination_analysis(df_data, df_foils, df_precursors)
    # Plot temperature curve
    plot_temperatures(
        df_results, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    # Plot LCA over time for single edge
    plot_LCA_change(
        df_results, 
        metal=metal, 
        precursor_suffix=precursor_suffix, 
        x_axis=x_axis, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    # Plot LCA over time for all edges
    plot_reduction_comparison(
        df_results, 
        precursor_type='all', 
        x_axis=x_axis, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    return None

def in_situ_analysis_averages(
    data_paths: Union[str, list[str]],
    synchrotron: str,
    measurements_to_foil: Union[int, list, np.ndarray, range],
    measurements_to_precursor: Union[int, list, np.ndarray, range],
    metal: str,
    precursor_suffix: str,
    x_axis: str='Measurement',
    file_selection_condition: str='*',
    negated_condition: bool=False,
    use_preedge: bool=True,
    use_transmission: bool=False,
    interactive: bool=False,
    with_uncertainty: bool=True,
) -> None:
    """Function that runs the entire analysis pipeline for in-situ analysis with measured standards.

    Args:
        data_paths (Union[str, list[str]]): Path or list of paths to folders containing the measured data.
        synchrotron (str): Name of the synchrotron the data was measured at.
        measurements_to_foil (Union[int, list, np.ndarray, range]): The number of measurements or measurement indeces to average when creating a reference spectra for the reduced metal. 
        measurements_to_precursor (Union[int, list, np.ndarray, range]): The number of measurements or measurement indeces to average when creating a reference spectra for the unreduced precursor.
        metal (str): The metal to use for plotting.
        precursor_suffix (str): The precursor to use for plotting.
        x_axis (str, optional): The column to use for the x-axis when plotting. Defaults to 'Measurement'.
        file_selection_condition (str, optional): Pattern to match with filenames. Defaults to '*'.
        negated_condition (bool, optional): Whether the filenames matching the pattern is included or excluded. Defaults to False.
        use_preedge (bool, optional): Boolean flag controlling if the pre-edge fit is subtracted during normalization. Defaults to True.
        use_transmission (bool, optional): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
        interactive (bool, optional): Boolean flag deciding if plots are interactive or static. Defaults to False.
        with_uncertainty (bool, optional): Boolean flag deciding if the uncertainties are plotted. Defaults to True.

    Returns:
        None
    """
    # Clears the notebook cell output when this function is called.
    clear_output(wait=True)
    # Read the in-situ data
    if type(data_paths) == list and len(data_paths) > 1:
        # Create empty list to hold all datasets
        list_of_datasets = []
        # Load data
        for path in data_paths:
            df_data = load_xas_data(
                path, 
                synchrotron=synchrotron, 
                file_selection_condition=file_selection_condition, 
                negated_condition=negated_condition, 
                verbose=False,
            )
            # Initial data processing
            df_data = processing_df(df_data, synchrotron=synchrotron, metal=metal)
            # Append to list of datasets
            list_of_datasets.append(df_data)
        # Combine the datasets
        df_data = combine_datasets(list_of_datasets)
    else:
        if type(data_paths) == list:
            data_paths = data_paths[0]
        # Load data
        df_data = load_xas_data(
            data_paths, 
            synchrotron=synchrotron, 
            file_selection_condition=file_selection_condition, 
            negated_condition=negated_condition,
        )
        # Initial data processing
        df_data = processing_df(df_data, synchrotron=synchrotron, metal=metal)
    # Convert number of measurements to use into range over the last measurements in the data
    if type(measurements_to_foil) == int:
        n_measurements = np.amax(df_data['Measurement'].unique()) + 1
        measurements_to_foil = range(n_measurements - measurements_to_foil, n_measurements)
    # Create dataframe with the reference spectra for reduced metals
    df_foils = average_measurements(df_data, measurements_to_foil)
    # Calculate the edge energy shift at each edge
    edge_correction_energies = {
    'Pd':calc_edge_correction(df_foils, metal='Pd', edge='K', transmission=use_transmission),
    'Ag':calc_edge_correction(df_foils, metal='Ag', edge='K', transmission=use_transmission),
    'Rh':calc_edge_correction(df_foils, metal='Rh', edge='K', transmission=use_transmission),
    'Ru':calc_edge_correction(df_foils, metal='Ru', edge='K', transmission=use_transmission),
    'Mn':calc_edge_correction(df_foils, metal='Mn', edge='K', transmission=use_transmission),
    'Mo':calc_edge_correction(df_foils, metal='Mo', edge='K', transmission=use_transmission),
    'Ir':calc_edge_correction(df_foils, metal='Ir', edge='L3', transmission=use_transmission),
    'Pt':calc_edge_correction(df_foils, metal='Pt', edge='L3', transmission=use_transmission),
    }
    # Normalization of the foil average
    normalize_data(
        df_foils, 
        edge_correction_energies, 
        subtract_preedge=use_preedge, 
        transmission=use_transmission,
    )
    print(f'Reference spectra from measurement {measurements_to_foil} created.')
    # Convert number of measurements to use into range over the first measurements in the data
    if type(measurements_to_precursor) == int:
        measurements_to_precursor = range(0, measurements_to_precursor)
    # Create dataframe with the reference spectra for unreduced precursors
    df_precursors = average_measurements(df_data, measurements_to_precursor)
    # Normalization of the precursor average
    normalize_data(
        df_precursors, 
        edge_correction_energies, 
        subtract_preedge=use_preedge, 
        transmission=use_transmission,
    )
    print(f'Reference spectra from measurement {measurements_to_precursor} created.')
    # Normalization of the data
    normalize_data(
        df_data, 
        edge_correction_energies, 
        subtract_preedge=use_preedge, 
        transmission=use_transmission
    )
    # Plotting of normalized data
    plot_data(
        df_data, 
        metal=metal, 
        foils=df_foils, 
        precursors=df_precursors, 
        precursor_suffix=precursor_suffix, 
        interactive=interactive,
    )
    # Linear combination analysis (LCA)
    df_results = linear_combination_analysis(df_data, df_foils, df_precursors)
    # Plot temperature curve
    plot_temperatures(
        df_results, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    # Plot LCA over time for single edge
    plot_LCA_change(
        df_results, 
        metal=metal, 
        precursor_suffix=precursor_suffix, 
        x_axis=x_axis, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    # Plot LCA over time for all edges
    plot_reduction_comparison(
        df_results, 
        precursor_type='all', 
        x_axis=x_axis, 
        with_uncertainty=with_uncertainty, 
        interactive=interactive
    )
    return None

class EventHandler(PatternMatchingEventHandler):
    def __init__(
        self, 
        function: Any,
        function_arguments: dict,
        patterns: Union[list[str], None]=None, 
        ignore_patterns: Union[list[str], None]=None, 
        ignore_directories: bool=False, 
        case_sensitive: bool=False,
        sleep_time: Union[int,float]=1,
        ) -> None:
        """Class instance that defines how events are handled.

        Args:
            function (Any): The function to run when a change happens.
            function_arguments (dict): The arguments parsed to the function.
            patterns (Union[list[str], None], optional): Pattern included in the filenames of watched files. Defaults to None.
            ignore_patterns (Union[list[str], None], optional): Pattern included in the filenames of ignored files. Defaults to None.
            ignore_directories (bool, optional): Whether changes to directories are ignored. Defaults to False.
            case_sensitive (bool, optional): Whether the filesystem is case sensitive. Defaults to False.
            sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.
        """        
        # Pass arguments to parent class
        super().__init__(
            patterns, 
            ignore_patterns, 
            ignore_directories, 
            case_sensitive
        )
        # Assign given arguments
        self.function = function
        self.function_arguments = function_arguments
        self.sleep_time = sleep_time
    
    # Functions describing the action to take when an event happens in the observed directory
    # Call a function when a file is modified
    # def on_modified(self, event) -> None:
    #     """Executes some action when a file is modified.

    #     Args:
    #         event: event instance.

    #     Returns:
    #         None
    #     """        
    #     self.function(**self.function_arguments)
    #     time.sleep(self.sleep_time)
    #     return None
    # Call a function when a file is created
    def on_created(self, event) -> None:
        """Executes some action when a file is created.

        Args:
            event: event instance.

        Returns:
            None
        """        
        self.function(**self.function_arguments)
        time.sleep(self.sleep_time)
        return None
    # Call a function when any event happens
    # def on_any_event(self, event) -> None:
    #     """Executes some action when any event happens.

    #     Args:
    #         event: event instance.

    #     Returns:
    #         None
    #     """        
    #     self.function(**self.function_arguments)
    #     time.sleep(self.sleep_time)
    #     return None

def watch_insitu_experiment(
    data_path: str,
    function: Any,
    function_arguments: dict,
    patterns: Union[list[str], None]=['*'],
    ignore_patterns: Union[list[str], None]=None,
    ignore_directories: bool=False,
    case_sensitive: bool=False,
    recursive: bool=True,
    sleep_time: Union[int,float]=1,
) -> None:
    """Continuously watch a directory for changes and re-run analysis code when a relevant change happens.

    Args:
        data_path (str): Path to the directory to watch for changes.
        function (Any): The function to run when a change happens.
        function_arguments (dict): The arguments parsed to the function.
        patterns (Union[list[str], None], optional): Pattern included in the filenames of watched files. Defaults to ['*'].
        ignore_patterns (Union[list[str], None], optional): Pattern included in the filenames of ignored files. Defaults to None.
        ignore_directories (bool, optional): Whether changes to directories are ignored. Defaults to False.
        case_sensitive (bool, optional): Whether the filesystem is case sensitive. Defaults to False.
        recursive (bool, optional): True if events will be emitted for sub-directories traversed recursively; False otherwise. Defaults to True.
        sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.

    Returns:
        None
    """
    # Create event handler
    event_handler = EventHandler(
        function=function,
        function_arguments=function_arguments,
        patterns=patterns, 
        ignore_patterns=ignore_patterns, 
        ignore_directories=ignore_directories, 
        case_sensitive=case_sensitive,
        sleep_time=sleep_time,
    )
    # Create observer
    observer = Observer()
    # Tell the observer where to look and what to do
    observer.schedule(
        event_handler=event_handler,
        path=data_path,
        recursive=recursive
    )
    # Start observing
    observer.start()
    try:
        while True:
            time.sleep(sleep_time)
    # Ensure proper shutdown of the observer
    except KeyboardInterrupt:
        observer.stop()
        print(f'\nNo longer observing {data_path}')
    observer.join()
    return None

def watch_insitu_experiment_v2(
    data_path: str,
    function: Any,
    function_arguments: dict,
    pattern: str='*',
    sleep_time: Union[int,float]=1,
) -> None:
    """Continuously watch a directory for changes and re-run analysis code when a relevant change happens.

    Args:
        data_path (str): Path to the directory to watch for changes.
        function (Any): The function to run when a change happens.
        function_arguments (dict): The arguments parsed to the function.
        pattern (str, optional): Pattern included in the filenames of watched files. Defaults to ['*'].
        sleep_time (Union[int,float], optional): Time in seconds before a change can cause an update of the analysis. Defaults to 1 second.

    Returns:
        None
    """
    # Tell the observer where to look and what to do
    folder_to_watch = Path(data_path)
    # Start observing
    last_change = None
    try:
        # Continuously watch for changes
        while True:
            # Find the most recently modified file
            latest_path = max(folder_to_watch.glob(pattern), key=lambda path: path.stat().st_mtime)
            # Get the time of modification
            latest_change = latest_path.stat().st_mtime
            # Check if the modification is new
            if latest_change != last_change:
                # Call function upon change
                function(**function_arguments)
                # Save the time of modification
                last_change = latest_change
            # Wait before looking for changes again
            time.sleep(sleep_time)
    # Ensure proper shutdown of the observer
    except KeyboardInterrupt:
        print(f'\nNo longer observing {data_path}')
    return None