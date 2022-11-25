#%% Imports

# Packages for math
import numpy as np
# Packages for typing
from typing import Union
# Packages for handling data
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# from tqdm.auto import tqdm # Use as standard. If progress bar is not rendering use normal tqdm below.
from tqdm import tqdm
# Packages for fitting
from lmfit import Parameters, fit_report, minimize
from lmfit.minimizer import MinimizerResult

#%% Fitting functions

def linear_combination(
    weights: Parameters, 
    basis_functions: list[np.array],
) -> np.array:
    """------------------------------------------------------------------
    Calculates the linear combination of a set of basis functions.

    Args:
        weights (Parameters): The weights of the basis functions.
        basis_functions (list[np.array]): The basis functions.

    Returns:
        np.array: The linear combination.
    """    
    # Unpack the weights from the Parameter instance
    weights = list(weights.valuesdict().values()) 
    # Create zero array for the linear combination
    combination = np.zeros_like(basis_functions[0])
    # Create variable to ensure sum of weights is 1
    weight_sum = 0
    # Loop to add the contributions to the linear combination together
    # List of basis functions should be 1 element longer than the weights
    for weight, basis_function in zip(weights, basis_functions):#[:-1]): 
        # Addition of the individual contribution to the linear combination
        combination += weight * basis_function
        # Sum of the weights
        weight_sum += weight
    return combination

def residual(
    target: np.array, 
    estimate: np.array
) -> np.array:
    """------------------------------------------------------------------
    Calculate residual between to arrays.

    Args:
        target (np.array): The target array.
        estimate (np.array): The estimated array.

    Returns:
        np.array: The residual array.
    """    
    return target - estimate

def fit_func(
    weights: Parameters, 
    basis_functions: list[np.array], 
    data_to_fit: np.array
) -> np.array:
    """------------------------------------------------------------------
    Fitting function used by lmfit.minimize for fitting the weights in Linear Combination Analysis (LCA).

    Args:
        weights (Parameters): The weights of the basis functions.
        basis_functions (list[np.array]): The basis functions.
        data_to_fit (np.array): The measurement to be fitted by LCA.

    Returns:
        np.array: The residual between the measured data and the linear combination.
    """    
    # Calculate the linear combination of the basis functions
    estimate = linear_combination(weights, basis_functions)
    # Calculate the residual
    return residual(data_to_fit, estimate)

def linear_combination_analysis(
    data: pd.DataFrame, 
    products: pd.DataFrame, 
    precursors: pd.DataFrame, 
    intermediates: Union[pd.DataFrame, None]=None,
    fit_min: Union[float, int, None]=0,
    fit_max: Union[float, int, None]=np.infty,
    verbose: bool=False, 
    return_dataframe: bool=True
) -> Union[pd.DataFrame, list[MinimizerResult]]:
    """------------------------------------------------------------------
    Linear Combination Analysis (LCA) of an entire dataset. 

    Provided with a dataset and standards of the relevant metal foils and precursors LCA is performed for all combinations of metals and their precursors on each measurement.

    Args:
        data (pd.DataFrame): The normalized data.
        products (pd.DataFrame): The normalized product standards.
        precursors (pd.DataFrame): The normalized precursor standards.
        intermediate (optional, pd.DataFrame): The normalized intermediate standards.
        fit_min (optional, Union[float, int, None]): The minimum energy value to include in the LCA fit. Defaults to 0.
        fit_max (optional, Union[float, int, None]): The maximum energy value to include in the LCA fit. Defaults to infinity.
        verbose (optional, bool): Boolean flag controlling if the fit results are printed. Defaults to False.
        return_dataframe (optional, bool): Boolean flag controlling if a dataframe (True) or a list of fit result objects (False) are returned. Defaults to True.

    Returns:
        pd.DataFrame | list[MinimizerResult]: The results of the LCA on the dataset as either a dataframe or a list of MinimizerResult objects.
    """    
    # List to hold lmfit output objects
    fit_results = []
    # Lists to hold values for Dataframe
    list_experiments = []
    list_metals = []
    list_products = []
    list_intermediates = []
    list_precursors = []
    list_precursor_types = []
    list_measurements = []
    list_temperatures = []
    list_parameters = []
    list_values = []
    list_stderrors = []
    list_stderrors_corrected = []
    list_energy_range = []
    list_basis_functions = []
    # Calculate the number of combinations to perform LCA for
    metals = data['Metal'].unique()
    n_combinations = 0
    for metal in metals:
        if intermediates == None:
            n_combinations += len(precursors['Experiment'][precursors['Metal'] == metal].unique()) * len(products['Experiment'][products['Metal'] == metal].unique())
        else:
            n_combinations += len(precursors['Experiment'][precursors['Metal'] == metal].unique()) * len(products['Experiment'][products['Metal'] == metal].unique()) * (len(intermediates['Experiment'][products['Metal'] == metal].unique()) + 1)
    # Progress bar for the LCA progress
    with tqdm(data['Metal'].unique(), total=n_combinations, desc='LCA progress: ') as pbar_metal:
        # Loop over all metal edges
        for metal in data['Metal'].unique():
            # Loop over all relevant precursors
            relevant_precursors = precursors['Experiment'][precursors['Metal'] == metal].unique()
            for precursor in relevant_precursors:
                # Loop over all relevant products
                relevant_products = products['Experiment'][products['Metal'] == metal].unique()
                for product in relevant_products:
                    # Initialize intermediate as None
                    intermediate = None
                    # Update descriptive text on progress bar
                    pbar_metal.set_postfix_str(f'Analysing {precursor} + {product}')
                    # Loop over all measurements
                    with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'{product} + {precursor}', mininterval=0.01) as pbar_measurement:
                        for measurement in data['Measurement'][data['Metal'] == metal].unique():
                            # Check if the metal exists in the dataset
                            # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
                            # Create filter for relevant values
                            data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
                            # Extract the energy range covered by a measurement
                            data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
                            # Extract the relevant data
                            data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)]
                            # Extract the temperature
                            temperature = np.median(data['Temperature'][data_filter])
                            # Check if the metal foil exists in the dataset
                            assert product in  products['Experiment'].unique(), f'No metal foil standard with the name: {product}\n\nValid values are: {products.Experiment.unique()}'
                            # Create filter for relevant values
                            product_filter = (products['Metal'] == metal) & (products['Measurement'] == 1)
                            # Extract the relevant data
                            product_basis = np.interp(data_range, products['Energy_Corrected'][product_filter], products['Normalized'][product_filter])
                            # Check if the precursor exists in the dataset
                            assert precursor in precursors['Experiment'].unique(), f'No precursor standard with the name: {precursor}\n\nValid values are: {precursors.Experiment.unique()}'
                            # Create filter for relevant values
                            precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
                            # Extract the relevant data
                            precursor_basis = np.interp(data_range, precursors['Energy_Corrected'][precursor_filter], precursors['Normalized'][precursor_filter])
                            # Group the two basis functions
                            basis_functions = [product_basis, precursor_basis]
                            # Initialize the fit parameters
                            fit_params = Parameters()
                            fit_params.add('product_weight', value=0.5, min=0, max=1)
                            fit_params.add('precursor_weight', expr='1.0 - product_weight', min=0, max=1)
                            try:
                                # Fit the linear combination of basis functions to the measurement
                                fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
                                # Store the fit results as lmfit MinimizerResults
                                fit_results.append(fit_out)
                            except:
                                raise Exception(f'Error occurred when fitting measurement {measurement} of {precursor} + {product}.')
                            # Save the values needed for the results Dataframe
                            for name, param in fit_out.params.items():
                                list_experiments.append(f'{precursor} + {product}')
                                list_metals.append(metal)
                                list_products.append(product)
                                list_intermediates.append(intermediate)
                                list_precursors.append(precursor)
                                list_precursor_types.append(precursor[2:])
                                list_measurements.append(measurement)
                                list_temperatures.append(temperature)
                                list_parameters.append(name)
                                list_values.append(param.value)
                                list_stderrors.append(param.stderr)
                                list_energy_range.append(data_range)
                                # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
                                if name == 'precursor_weight' and param.value >= 0.999:
                                    list_stderrors_corrected.append(list_stderrors[-2])
                                else:
                                    list_stderrors_corrected.append(param.stderr)
                            list_basis_functions.extend(basis_functions)
                            # Check if the fit results should be printed
                            if verbose:
                                print(f'Fit results for the linear combination of {product} and {precursor} to measurement {measurement} of the {metal} edge.\n')
                                print(fit_report(fit_out))
                                print('\n')
                    # Update pbar
                    pbar_metal.update()
                    if intermediates != None:
                        # Loop over all relevant intermediates
                        relevant_intermediates = intermediates['Experiment'][intermediates['Metal'] == metal].unique()
                        for intermediate in relevant_intermediates:
                            # Update descriptive text on progress bar
                            pbar_metal.set_postfix_str(f'Analysing {precursor} + {product} + {intermediate}')
                            # Loop over all measurements
                            with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'{precursor} + {product} + {intermediate}', mininterval=0.01) as pbar_measurement:
                                for measurement in pbar_measurement:
                                    # Check if the metal exists in the dataset
                                    # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
                                    # Create filter for relevant values
                                    data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
                                    # Extract the energy range covered by a measurement
                                    data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
                                    # Extract the relevant data
                                    data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)]
                                    # Extract the temperature
                                    temperature = np.median(data['Temperature'][data_filter])
                                    # Check if the metal foil exists in the dataset
                                    assert product in products['Experiment'].unique(), f'No product standard with the name: {product}\n\nValid values are: {products.Experiment.unique()}'
                                    # Create filter for relevant values
                                    product_filter = (products['Experiment'].str.contains(metal)) & (products['Measurement'] == 1)
                                    # Extract the relevant data
                                    product_basis = np.interp(data_range, products['Energy_Corrected'][product_filter], products['Normalized'][product_filter])
                                    # Check if the intermediate exists in the dataset
                                    assert intermediate in intermediates['Experiment'].unique(), f'No precursor standard with the name: {intermediate}\n\nValid values are: {intermediates.Experiment.unique()}'
                                    # Create filter for relevant values
                                    intermediate_filter = (intermediates['Experiment'] == intermediate) & (intermediates['Measurement'] == 1)
                                    # Extract the relevant data
                                    intermediate_basis = np.interp(data_range, intermediates['Energy_Corrected'][intermediate_filter], intermediates['Normalized'][intermediate_filter])
                                    # Check if the precursor exists in the dataset
                                    assert precursor in precursors['Experiment'].unique(), f'No precursor standard with the name: {precursor}\n\nValid values are: {precursors.Experiment.unique()}'
                                    # Create filter for relevant values
                                    precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
                                    # Extract the relevant data
                                    precursor_basis = np.interp(data_range, precursors['Energy_Corrected'][precursor_filter], precursors['Normalized'][precursor_filter])
                                    # Group the two basis functions
                                    basis_functions = [product_basis, intermediate_basis, precursor_basis]
                                    # Initialize the fit parameters
                                    fit_params = Parameters()
                                    fit_params.add('product_weight', value=0.33, min=0, max=1)
                                    fit_params.add('intermediate_weight', value=0.33, min=0, max=1)
                                    fit_params.add('precursor_weight', expr='1.0 - product_weight - intermediate_weight', min=0, max=1)
                                    try:
                                        # Fit the linear combination of basis functions to the measurement
                                        fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
                                        # Store the fit results as lmfit MinimizerResults
                                        fit_results.append(fit_out)
                                    except:
                                        raise Exception(f'Error occurred when fitting measurement {measurement} of {precursor} + {product} + {intermediate}.')
                                    # Save the values needed for the results Dataframe
                                    for name, param in fit_out.params.items():
                                        list_experiments.append(f'{precursor} + {intermediate} + {product}')
                                        list_metals.append(metal)
                                        list_products.append(product)
                                        list_intermediates.append(intermediate)
                                        list_precursors.append(precursor)
                                        list_precursor_types.append(precursor[2:])
                                        list_measurements.append(measurement)
                                        list_temperatures.append(temperature)
                                        list_parameters.append(name)
                                        list_values.append(param.value)
                                        list_stderrors.append(param.stderr)
                                        list_energy_range.append(data_range)
                                        # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to themaximum allowed values
                                        if name == 'precursor_weight' and param.value >= 0.999:
                                            list_stderrors_corrected.append(list_stderrors[-2])
                                        else:
                                            list_stderrors_corrected.append(param.stderr)
                                    list_basis_functions.extend(basis_functions)
                                    # Check if the fit results should be printed
                                    if verbose:
                                        print(f'Fit results for the linear combination of {product}, {intermediate} and {precursor} to measurement {measurement} of the {metal} edge.\n')
                                        print(fit_report(fit_out))
                                        print('\n')
                            # Update pbar
                            pbar_metal.update()
    # Save the fit results as a Dataframe
    df_results = pd.DataFrame({
        'Experiment':list_experiments,
        'Metal':list_metals,
        'Product':list_products,
        'Intermediate':list_intermediates,
        'Precursor':list_precursors,
        'Precursor Type': list_precursor_types,
        'Measurement':list_measurements,
        'Temperature':list_temperatures,
        'Temperature Average': 0,
        'Temperature Std': 0,
        'Parameter':list_parameters,
        'Value':list_values,
        'StdErr':list_stderrors,
        'StdCorrected':list_stderrors_corrected,
        'Energy Range': list_energy_range,
        'Basis Function':list_basis_functions,
        })
    # Calculate average and standard deviation of the temperature across all edges within 1 measurement
    for measurement in df_results['Measurement'].unique():
        # Create filter for relvant values
        df_filter = (df_results['Parameter'] == 'product_weight') & (df_results['Precursor Type'] == df_results['Precursor Type'].mode().to_list()[0])
        meas_filter = (df_results['Measurement'] == measurement)
        # Calculate the mean
        avg_temp = np.mean(df_results['Temperature'][df_filter & meas_filter])
        df_results['Temperature Average'][meas_filter] = avg_temp 
        # Calculate the standard deviation
        std_temp = np.std(df_results['Temperature'][df_filter & meas_filter], ddof=1)
        df_results['Temperature Std'][meas_filter] = std_temp 
    # Replace NaN values with interpolated values
    df_results = df_results.interpolate()
    # Check if the dataframe or lmfit MinimizerResults should be returned
    if return_dataframe:
        return df_results
    else:
        return fit_results

def LCA_internal(
    data: pd.DataFrame, 
    initial_state_index: int=0, 
    final_state_index: int=-1, 
    intermediate_state_index: Union[int, None]=None,
    fit_min: Union[float, int]=0,
    fit_max: Union[float, int]=np.inf,
    verbose: bool=False, 
    return_dataframe: bool=True
) -> Union[pd.DataFrame, list[MinimizerResult]]:
    """------------------------------------------------------------------
    Linear Combination Analysis (LCA) of an experiment using measurements as the components. 

    Provided with a dataset and standards of the relevant metal foils and precursors LCA is performed for all combinations of metals and their precursors on each measurement.

    Args:
        data (pd.DataFrame): The normalized data.
        initial_state_index (int): The measurement to use as a reference for the initial state.
        final_state_index (int): The measurement to use as a reference for the final state.
        intermediate_state_index (optional, Union[int, None]): The measurement to use as a reference for the intermediate state. Defaults to None.
        fit_min (optional, Union[float, int, None]): The minimum energy value to include in the LCA fit. Defaults to 0.
        fit_max (optional, Union[float, int, None]): The maximum energy value to include in the LCA fit. Defaults to infinity.
        verbose (optional, bool): Boolean flag controlling if the fit results are printed. Defaults to False.
        return_dataframe (optional, bool): Boolean flag controlling if a dataframe (True) or a list of fit result objects (False) are returned. Defaults to True.

    Returns:
        pd.DataFrame | list[MinimizerResult]: The results of the LCA on the dataset as either a dataframe or a list of MinimizerResult objects.
    """    
    # List to hold lmfit output objects
    fit_results = []
    # Lists to hold values for Dataframe
    list_experiments = []
    list_metals = []
    list_products = []
    list_intermediates = []
    list_precursors = []
    list_precursor_types = []
    list_measurements = []
    list_temperatures = []
    list_parameters = []
    list_values = []
    list_stderrors = []
    list_stderrors_corrected = []
    list_energy_range = []
    list_basis_functions = []
    if final_state_index == -1:
        final_state_name = 'last'
        final_state_index = np.amax(data['Measurement'])
    else:
        final_state_name = final_state_index
    # Create progress bar for LCA progress
    with tqdm(data['Metal'].unique(), desc='LCA progress: ') as pbar_metal:
        # Loop over all metal edges
        for metal in data['Metal'].unique():
            if not intermediate_state_index:
                # Update descriptive text on progress bar
                pbar_metal.set_postfix_str(f'Analysing frame {initial_state_index} + {final_state_index}')
                # Loop over all measurements
                with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'Frame {initial_state_index} + {final_state_index}', mininterval=0.01) as pbar_measurement:
                    for measurement in data['Measurement'][data['Metal'] == metal].unique():
                        # Check if the metal exists in the dataset
                        # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
                        # Create filter for relevant values
                        data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
                        # Extract the energy range covered by a measurement
                        data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
                        # Extract the relevant data
                        data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)].to_numpy()
                        # Extract the temperature
                        temperature = np.median(data['Temperature'][data_filter])
                        # Select the reference data
                        initial_filter = (data['Metal'] == metal) & (data['Measurement'] == initial_state_index)
                        initial_basis = np.interp(data_range, data['Energy_Corrected'][initial_filter], data['Normalized'][initial_filter])
                        final_filter = (data['Metal'] == metal) & (data['Measurement'] == final_state_index)
                        final_basis = np.interp(data_range, data['Energy_Corrected'][final_filter], data['Normalized'][final_filter])
                        # Group the two basis functions
                        basis_functions = [final_basis, initial_basis]
                        # Initialize the fit parameters
                        fit_params = Parameters()
                        fit_params.add('product_weight', value=0.5, min=0, max=1)
                        fit_params.add('precursor_weight', expr='1.0 - product_weight', min=0, max=1)
                        try:
                            # Fit the linear combination of basis functions to the measurement
                            fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
                            # Store the fit results as lmfit MinimizerResults
                            fit_results.append(fit_out)
                        except:
                            raise Exception(f'Error occurred when fitting measurement {measurement} with frame {initial_state_index} + {final_state_index}.')
                        # Save the values needed for the results Dataframe
                        for name, param in fit_out.params.items():
                            list_experiments.append(f'Frame {initial_state_index} + {final_state_name}')
                            list_metals.append(metal)
                            list_products.append(final_state_name)
                            list_intermediates.append(intermediate_state_index)
                            list_precursors.append(initial_state_index)
                            list_precursor_types.append('Internal')
                            list_measurements.append(measurement)
                            list_temperatures.append(temperature)
                            list_parameters.append(name)
                            list_values.append(param.value)
                            list_stderrors.append(param.stderr)
                            list_energy_range.append(data_range)
                            # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
                            if not param.stderr:
                                param.stderr = 0
                            if name == 'precursor_weight' and (param.value >= 0.99 or param.stderr > 1.):
                                list_stderrors_corrected.append(list_stderrors[-2])
                            else:
                                list_stderrors_corrected.append(param.stderr)
                        list_basis_functions.extend(basis_functions)
                        # Check if the fit results should be printed
                        if verbose:
                            print(f'Fit results for the linear combination of frame {initial_state_index} and {final_state_index} to measurement {measurement} of the {metal} edge.\n')
                            print(fit_report(fit_out))
                            print('\n')
                        # Update pbar
                        pbar_metal.update()
            elif intermediate_state_index:
                # Update descriptive text on progress bar
                pbar_metal.set_postfix_str(f'Analysing frame {initial_state_index} + {intermediate_state_index} + {final_state_index}')
                # Loop over all measurements
                with tqdm(data['Measurement'][data['Metal'] == metal].unique(), leave=False, desc=f'Frame {initial_state_index} + {intermediate_state_index} + {final_state_index}', mininterval=0.01) as pbar_measurement:
                    for measurement in pbar_measurement:
                        # Check if the metal exists in the dataset
                        # assert data['Experiment'].str.contains(metal).any(), f'No experiment with the name: {metal}\n\nValid values are: {data.Experiment.unique()}'
                        # Create filter for relevant values
                        data_filter = (data['Metal'] == metal) & (data['Measurement'] == measurement)
                        # Extract the energy range covered by a measurement
                        data_range = data['Energy_Corrected'][data_filter & (data['Energy_Corrected'] >= fit_min) & (data['Energy_Corrected'] <= fit_max)].to_numpy()
                        # Extract the relevant data
                        data_to_fit = data['Normalized'][data_filter & data['Energy_Corrected'].isin(data_range)].to_numpy()
                        # Extract the temperature
                        temperature = np.median(data['Temperature'][data_filter])
                        # Select the reference data
                        initial_filter = (data['Metal'] == metal) & (data['Measurement'] == initial_state_index)
                        initial_basis = np.interp(data_range, data['Energy_Corrected'][initial_filter], data['Normalized'][initial_filter])
                        intermediate_filter = (data['Metal'] == metal) & (data['Measurement'] == intermediate_state_index)
                        intermediate_basis = np.interp(data_range, data['Energy_Corrected'][intermediate_filter], data['Normalized'][intermediate_filter])
                        final_filter = (data['Metal'] == metal) & (data['Measurement'] == final_state_index)
                        final_basis = np.interp(data_range, data['Energy_Corrected'][final_filter], data['Normalized'][final_filter])
                        # Group the basis functions
                        basis_functions = [final_basis, intermediate_basis, initial_basis]
                        # Initialize the fit parameters
                        fit_params = Parameters()
                        fit_params.add('product_weight', value=0.33, min=0, max=1)
                        fit_params.add('intermediate_weight', value=0.33, min=0, max=1)
                        fit_params.add('precursor_weight', expr='1.0 - product_weight - intermediate_weight', min=0, max=1)
                        try:
                            # Fit the linear combination of basis functions to the measurement
                            fit_out = minimize(fit_func, fit_params, args=(basis_functions, data_to_fit,))
                            # Store the fit results as lmfit MinimizerResults
                            fit_results.append(fit_out)
                        except:
                            raise Exception(f'Error occurred when fitting measurement {measurement} with frame {initial_state_index} + {intermediate_state_index} + {final_state_index}.')
                        # Save the values needed for the results Dataframe
                        for name, param in fit_out.params.items():
                            list_experiments.append(f'Frame {initial_state_index} + {intermediate_state_index} + {final_state_name}')
                            list_metals.append(metal)
                            list_products.append(final_state_name)
                            list_intermediates.append(intermediate_state_index)
                            list_precursors.append(initial_state_index)
                            list_precursor_types.append('Internal')
                            list_measurements.append(measurement)
                            list_temperatures.append(temperature)
                            list_parameters.append(name)
                            list_values.append(param.value)
                            list_stderrors.append(param.stderr)
                            list_energy_range.append(data_range)
                            # Error estimation of the dependent parameter (precursor_weight) is inconsistent when close to the maximum allowed values
                            if name == 'precursor_weight' and (param.value >= 0.99 or param.stderr > 0.5):
                                list_stderrors_corrected.append(np.mean(list_stderrors_corrected[-2:]))
                            elif name == 'intermediate_weight' and (param.value <= 0.01 and param.stderr > 1.):
                                list_stderrors_corrected.append(np.mean(list_stderrors_corrected[-1]))
                            else:
                                list_stderrors_corrected.append(param.stderr)
                        list_basis_functions.extend(basis_functions)
                        # Check if the fit results should be printed
                        if verbose:
                            print(f'Fit results for the linear combination of frame {initial_state_index}, {intermediate_state_index} and {final_state_index} to measurement {measurement} of the {metal} edge.\n')
                            print(fit_report(fit_out))
                            print('\n')
                # Update pbar
                pbar_metal.update()
    # Save the fit results as a Dataframe
    df_results = pd.DataFrame({
        'Experiment':list_experiments,
        'Metal':list_metals,
        'Product':list_products,
        'Intermediate':list_intermediates,
        'Precursor':list_precursors,
        'Precursor Type': list_precursor_types,
        'Measurement':list_measurements,
        'Temperature':list_temperatures,
        'Temperature Average': 0,
        'Temperature Std': 0,
        'Parameter':list_parameters,
        'Value':list_values,
        'StdErr':list_stderrors,
        'StdCorrected':list_stderrors_corrected,
        'Energy Range': list_energy_range,
        'Basis Function':list_basis_functions,
        })
    # Calculate average and standard deviation of the temperature across all edges within 1 measurement
    for measurement in df_results['Measurement'].unique():
        # Create filter for relvant values
        df_filter = (df_results['Parameter'] == 'product_weight') & (df_results['Precursor Type'] == df_results['Precursor Type'].mode().to_list()[0])
        meas_filter = (df_results['Measurement'] == measurement)
        # Calculate the mean
        avg_temp = np.mean(df_results['Temperature'][df_filter & meas_filter])
        df_results['Temperature Average'][meas_filter] = avg_temp 
        # Calculate the standard deviation
        std_temp = np.std(df_results['Temperature'][df_filter & meas_filter], ddof=1)
        df_results['Temperature Std'][meas_filter] = std_temp 
    # Replace NaN values with interpolated values
    df_results = df_results.interpolate()
    # Check if the dataframe or lmfit MinimizerResults should be returned
    if return_dataframe:
        return df_results
    else:
        return fit_results