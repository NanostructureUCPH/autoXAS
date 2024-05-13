# %% Imports

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly_express as px
import seaborn as sns
import yaml
from larch import Group
from larch.xafs import find_e0, pre_edge
from larch.xray import xray_edge
from lmfit import Parameters, fit_report, minimize
from lmfit.minimizer import MinimizerResult
from tqdm.auto import tqdm
from sklearn.decomposition import NMF, PCA
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
sns.set_theme()
pio.renderers.default = 'notebook'

# %% Other functions

def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig

# %% autoXAS class

class autoXAS():
    def __init__(self, metals=None, edges=None) -> None:
        self.data_directory = None
        self.data_type = '.dat'
        self.data = None
        self.raw_data = None
        self.standards = None
        self.experiments = None
        self.save_directory = './'
        self.energy_column = None
        self.I0_columns = None
        self.I1_columns = None
        self.temperature_column = None
        self.metals = None
        self.xas_mode = 'Flourescence'
        self.energy_unit = 'eV'
        self.energy_column_unitConversion = 1
        self.temperature_unit = 'K'
        self.interactive = False
        self.edge_correction_energies = {}
        if metals and edges:
            self._calculate_edge_shift(metals, edges)
        
    
    def save_config(self, config_name: str, save_directory: str='./'):
        config = dict(
            data_directory=self.data_directory,
            data_type=self.data_type,
            energy_column=self.energy_column,
            I0_columns=self.I0_columns,
            I1_columns=self.I1_columns,
            temperature_column=self.temperature_column,
            edge_correction=self.edge_correction,
            xas_mode=self.xas_mode,
            energy_unit=self.energy_unit,
            energy_column_unitConversion=self.energy_column_unitConversion,
            temperature_unit=self.temperature_unit,
            save_directory=self.save_directory
        )
        with open(save_directory + config_name, 'w') as file:
            yaml.dump(config, file)
            
        return None
        
    def load_config(self, config_name: str, directory: str='./'):
        with open(directory + config_name, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.data_directory = config['data_directory']
        self.data_type = config['data_type']
        self.energy_column = config['energy_column']
        self.I0_columns = config['I0_columns']
        self.I1_columns = config['I1_columns']
        self.temperature_column = config['temperature_column']
        self.edge_correction = config['edge_correction']
        self.xas_mode = config['xas_mode']
        self.energy_unit = config['energy_unit']
        self.energy_column_unitConversion = config['energy_column_unitConversion']
        self.temperature_unit = config['temperature_unit']
        self.save_directory = config['save_directory']
        return None
    
    def _read_data(self):
        if self.data_directory is None:
            raise ValueError('No data directory specified')
        
        if self.data_type == '.dat':
            data_files = list(Path(self.data_directory).rglob('*.dat'))
            for file in tqdm(data_files, desc='Reading data files', leave=False):
                rows_to_skip = 0
                last_line = ''
                with open(file, 'r') as f:
                    for line in f:
                        if line.startswith('\n') or line.startswith('#'):
                            last_line = line
                            rows_to_skip += 1
                        else:
                            columns = last_line.split()[1:]
                            break
                
                raw_data = pd.read_csv(
                    file,
                    sep='\s+',
                    header=None,
                    names= columns,
                    skiprows=rows_to_skip,
                    skip_blank_lines=True,
                    on_bad_lines='skip',
                    keep_default_na=False,
                )
                
                raw_data = raw_data.apply(pd.to_numeric, errors='coerce')#, downcast='float')
                raw_data.dropna(inplace=True)
                
                data = pd.DataFrame()
                
                raw_data['File'] = file.name
                
                data['File'] = raw_data['File']
                data['Experiment'] = file.stem
                for fragment in file.stem.split('_'):
                    if len(fragment) < 3 and fragment.isalpha():
                        data['Metal'] = fragment
                
                # Calculate energy
                data['Energy'] = raw_data[self.energy_column] * self.energy_column_unitConversion
                
                # Calculate temperature
                if self.temperature_column is not None:
                    data['Temperature'] = raw_data[self.temperature_column]
                else:
                    data['Temperature'] = 0 # Placeholder for temperature
                
                # Calculate I0
                if isinstance(self.I0_columns, list):
                    data['I0'] = 0
                    for column in self.I0_columns:
                        data['I0'] += raw_data[column]
                elif isinstance(self.I0_columns, str):
                    data['I0'] = raw_data[self.I0_columns]
                # Calculate I1
                if isinstance(self.I1_columns, list):
                    data['I1'] = 0
                    for column in self.I1_columns:
                        data['I1'] += raw_data[column]
                elif isinstance(self.I1_columns, str):
                    data['I1'] = raw_data[self.I1_columns]
                
                # Calculate absorption coefficient
                if self.xas_mode == 'Flourescence':
                    data['mu'] = data['I1'] / data['I0']
                elif self.xas_mode == 'Transmission':
                    data['mu'] = np.log(data['I0'] / data['I1'])
                
                # Determine which measurement each data point belongs to
                measurement_number = 1
                measurement_number_values = []
                for energy_step in data['Energy'].diff().round(2):
                    if energy_step < 0:
                        measurement_number += 1
                    measurement_number_values.append(measurement_number)
                data['Measurement'] = measurement_number_values
                
                if self.raw_data is None:
                    self.raw_data = raw_data
                else:
                    self.raw_data = pd.concat([self.raw_data, raw_data])
                
                if self.data is None:
                    self.data = data
                else:
                    self.data = pd.concat([self.data, data]).reset_index(drop=True)
                    
        elif self.data_type == '.h5':
            raise NotImplementedError('HDF5 file reading not implemented yet')
        return None
    
    def load_standards(self, standards_directory: str, standards_type: str='.dat'):
        raise NotImplementedError('Standard loading not implemented yet')
    
    def _calculate_edge_shift(self, metals: list[str], edges: list[str]):
        for metal, edge in zip(metals, edges):
            edge_energy_table = xray_edge(metal, edge, energy_only=True)
            
            measurement_filter = (self.data['Metal'] == metal) & (self.data['Experiment'] == 1)
            edge_energy_measured = find_e0(self.data['Energy'][measurement_filter], self.data['mu'][measurement_filter])
            self.edge_correction_energies[metal] = edge_energy_table - edge_energy_measured
        return None
    
    def _energy_correction(self):
        for experiment in tqdm(self.experiments, desc='Energy correction', leave=False):
            experiment_filter = (self.data['Experiment'] == experiment)
            n_measurements = self.data['Measurement'][experiment_filter].max()
            # Correct for small variations in measured energy points
            energy = self.data['Energy'][experiment_filter].to_numpy().reshape(n_measurements, -1)
            energy_correction = energy.mean(axis=0)
            # Correct for edge shift
            energy_correction += self.edge_correction_energies.get(self.data['Metal'][experiment_filter].values[0], 0.0)
            # Estimate mu at corrected energy points using linear interpolation
            for measurement in range(1, n_measurements+1):
                measurement_filter = (self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)
                i0_interpolated = np.interp(energy_correction, self.data['Energy'][measurement_filter], self.data['I0'][measurement_filter])
                i1_interpolated = np.interp(energy_correction, self.data['Energy'][measurement_filter], self.data['I1'][measurement_filter])
                mu_interpolated = np.interp(energy_correction, self.data['Energy'][measurement_filter], self.data['mu'][measurement_filter])
                
                # Apply correction
                self.data['Energy'][measurement_filter] = energy_correction
                self.data['I0'][measurement_filter] = i0_interpolated
                self.data['I1'][measurement_filter] = i1_interpolated
                self.data['mu'][measurement_filter] = mu_interpolated
        return None
    
    def _average_data(self, measurements_to_average: Union[str, list[int], np.ndarray, range]='all'):
        avg_measurements = []
        for experiment in tqdm(self.experiments, desc='Averaging data', leave=False):
            first = True
            experiment_filter = (self.data['Experiment'] == experiment)
            
            if measurements_to_average == 'all':
                measurements = self.data['Measurement'][experiment_filter].unique()
            else:
                measurements = measurements_to_average
            
            for measurement in measurements:
                measurement_filter = (self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)
                
                if first:
                    df_avg = self.data[measurement_filter].copy()
                    i0_avg = np.zeros_like(df_avg['I0'], dtype=np.float64)
                    i1_avg = np.zeros_like(df_avg['I1'], dtype=np.float64)
                    mu_avg = np.zeros_like(df_avg['mu'], dtype=np.float64)
                    
                    first = False
                
                i0_avg += self.data['I0'][measurement_filter].to_numpy()
                i1_avg += self.data['I1'][measurement_filter].to_numpy()
                mu_avg += self.data['mu'][measurement_filter].to_numpy()
            
            n_measurements = len(measurements)
            df_avg['I0'] = i0_avg / n_measurements
            df_avg['I1'] = i1_avg / n_measurements
            df_avg['mu'] = mu_avg / n_measurements
            df_avg['Temperature'] = self.data['Temperature'][measurement_filter].mean()
            df_avg['Temperature (std)'] = self.data['Temperature'][measurement_filter].std()
            
            avg_measurements.append(df_avg)
            
        self.data = pd.concat(avg_measurements)
        return None
    
    def _average_data_periodic(self, period: Union[None, int]=None, n_periods: Union[None, int]=None):
        avg_measurements = []
        if (period and n_periods) or (not period and not n_periods) :
            n_arguments = bool(period) + bool(n_periods)
            raise Exception(f"Exactly 1 optional argument should be given. {n_arguments} was given.")
        for experiment in tqdm(self.experiments, desc='Averaging data', leave=False):
            experiment_filter = (self.data['Experiment'] == experiment)
            n_total_measurements = np.amax(self.data['Measurement'][experiment_filter])
            if period:
                n_measurements_to_average = period
                new_n_measurements = int(np.ceil(n_total_measurements / period))
            elif n_periods:
                n_measurements_to_average = n_total_measurements // n_periods
                new_n_measurements = n_periods
                
            measurements_to_average = np.arange(n_measurements_to_average)+1
            measurements_to_average_temp = measurements_to_average.copy()
            
            for measurement_number in range(new_n_measurements):
                if measurements_to_average_temp.any() >= n_total_measurements:
                    measurements_to_average_temp = np.array([i for i in measurements_to_average_temp if i < n_total_measurements])
                    
                for measurement in measurements_to_average_temp:
                    measurement_filter = (self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)
                    if measurement == measurements_to_average_temp[0]:
                        df_avg = self.data[measurement_filter].copy()
                        energy_avg = np.zeros_like(df_avg['Energy'], dtype=np.float64)
                        i0_avg = np.zeros_like(df_avg['I0'], dtype=np.float64)
                        i1_avg = np.zeros_like(df_avg['I1'], dtype=np.float64)
                        mu_avg = np.zeros_like(df_avg['mu'], dtype=np.float64)
                        
                    energy_avg += self.data['Energy'][measurement_filter].to_numpy()
                    i0_avg += self.data['I0'][measurement_filter].to_numpy()
                    i1_avg += self.data['I1'][measurement_filter].to_numpy()
                    mu_avg += self.data['mu'][measurement_filter].to_numpy()
                    
                n_measurements = len(measurements_to_average_temp)
                df_avg['Energy'] = energy_avg / n_measurements
                df_avg['I0'] = i0_avg / n_measurements
                df_avg['I1'] = i1_avg / n_measurements
                df_avg['mu'] = mu_avg / n_measurements
                df_avg['Temperature'] = self.data['Temperature'][measurement_filter].mean()
                df_avg['Temperature (std)'] = self.data['Temperature'][measurement_filter].std()
                df_avg['Measurement'] = measurement_number + 1
                
                measurements_to_average_temp += n_measurements_to_average
                
                avg_measurements.append(df_avg)
        self.data = pd.concat(avg_measurements)
        return None
    
    def _normalize_data(self):
        if self.data is None:
            raise ValueError('No data to normalize')
        
        self.data['mu_norm'] = 0
        self.data['pre_edge'] = 0
        self.data['post_edge'] = 0
        
        for experiment in tqdm(self.experiments, desc='Normalizing data', leave=False):
            experiment_filter = (self.data['Experiment'] == experiment)
            if self.edge_correction:
                raise NotImplementedError('Edge correction not implemented yet')
            for measurement in self.data['Measurement'][experiment_filter].unique():
                measurement_filter = (self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)
                dummy_group = Group(name='dummy')
                
                self.data['mu_norm'][measurement_filter] = self.data['mu'][measurement_filter] - np.amin(self.data['mu'][measurement_filter])

                try:
                    pre_edge(self.data['Energy'][measurement_filter], self.data['mu_norm'][measurement_filter], group=dummy_group, make_flat=False)
                    self.data['pre_edge'][measurement_filter] = dummy_group.pre_edge
                    self.data['post_edge'][measurement_filter] = dummy_group.post_edge
                    self.data['mu_norm'][measurement_filter] -= dummy_group.pre_edge
                    pre_edge(self.data['Energy'][measurement_filter], self.data['mu_norm'][measurement_filter], group=dummy_group, make_flat=False)
                    self.data['mu_norm'][measurement_filter] /= dummy_group.post_edge
                except:
                    self.data.drop(self.data[measurement_filter].index, inplace=True)
                    print(f'Error normalizing {experiment} measurement {measurement}. Measurement removed.')
        return None
    
    def load_data(self, average: Union[bool, str]=False, measurements_to_average: Union[str, list[int], np.ndarray, range]='all', n_periods: Union[None, int]=None, period: Union[None, int]=None):	
        self._read_data()
        self.experiments = list(self.data['Experiment'].unique())
        self.metals = list(self.data['Metal'].unique())
        self._energy_correction()
        if average:
            if average.lower() == 'standard':
                self._average_data(measurements_to_average=measurements_to_average)
            elif average.lower() == 'periodic':
                self._average_data_periodic(period=period, n_periods=n_periods)
            else:
                raise ValueError('Invalid average. Must be "standard" or "periodic".')
        self._normalize_data()
        return None
    
    def _linear_combination(self, weights: Parameters, components: list[np.array]) -> np.array:
        weights = np.array(list(weights.valuesdict().values()))
        components = np.array(components)
        return np.dot(weights, components)
        
    def _residual(self, target: np.array, combination: np.array) -> np.array:
        return target - combination
    
    def _fit_function(self, weights: Parameters, components: list[np.array], target: np.array) -> np.array:
        combination = self._linear_combination(weights, components)
        return self._residual(target, combination)
    
    # Analysis functions
    
    def LCA(self, use_standards: bool=False, components: Union[list[int], list[str], None]=[0,-1], fit_range: Union[None, tuple[float, float], list[tuple[float, float]]]=None):
        if use_standards and not self.standards:
            raise ValueError('No standards loaded')
        
        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError('Number of fit ranges must match number of metals')
        
        # Initialize list to store results
        list_experiment = []
        list_metal = []
        list_measurement = []
        list_temperature = []
        list_fit_range = []
        list_energy = []
        list_component_name = []
        list_component = []
        list_parameter_name = []
        list_parameter_value = []
        list_parameter_error = []
        
        for metal in tqdm(self.metals, desc='Performing LCA', leave=False):
            if use_standards:
                raise NotImplementedError('LCA with standards not implemented yet')
            else:
                component_measurements = self.data['Measurement'][self.data['Metal'] == metal].unique()[components]
                component_names = [f'Measurement {meas_num}' for meas_num in component_measurements]
                n_components = len(component_measurements)
                if n_components < 2:
                    raise ValueError('At least 2 components are required to perform LCA.')
                components_mu = []
                for measurement in component_measurements:
                    measurement_filter = (self.data['Metal'] == metal) & (self.data['Measurement'] == measurement)
                    components_mu.append(self.data['mu_norm'][measurement_filter].to_numpy())
                    
            if fit_range:
                raise NotImplementedError('Fit range not implemented yet')
            
            for measurement in self.data['Measurement'][self.data['Metal'] == metal].unique():
                measurement_filter = (self.data['Metal'] == metal) & (self.data['Measurement'] == measurement)
                target = self.data['mu_norm'][measurement_filter].to_numpy()
                
                weights = Parameters()
                for i in range(1,n_components):
                    weights.add(f'w{i}', value=1/n_components, min=0, max=1)
                weights.add(f'w{n_components}', min=0, max=1, expr='1 - ' + ' - '.join([f'w{i}' for i in range(1,n_components)]))	
                
                fit_output = minimize(self._fit_function, weights, args=(components_mu, target))

                # Store data used for LCA
                list_experiment.append(self.data['Experiment'][measurement_filter].values[0])
                list_metal.append(metal)
                list_measurement.append(measurement)
                list_temperature.append(self.data['Temperature'][measurement_filter].values[0])
                list_fit_range.append(fit_range)
                list_energy.append(self.data['Energy'][measurement_filter].to_numpy())
                list_component_name.append(f'(Ref) Measurement {measurement}')
                list_component.append(target)
                list_parameter_name.append(None)
                list_parameter_value.append(None)
                list_parameter_error.append(None)
                # Store LCA results
                for i, (name, param) in enumerate(fit_output.params.items()):
                    list_experiment.append(self.data['Experiment'][measurement_filter].values[0])
                    list_metal.append(metal)
                    list_measurement.append(measurement)
                    list_temperature.append(self.data['Temperature'][measurement_filter].values[0])
                    list_fit_range.append(fit_range)
                    list_energy.append(self.data['Energy'][measurement_filter].to_numpy())
                    list_component_name.append(component_names[i])
                    list_component.append(components_mu[i])
                    list_parameter_name.append(name)
                    list_parameter_value.append(param.value)
                    list_parameter_error.append(param.stderr)
                    
        self.LCA_result = pd.DataFrame({
            'Experiment': list_experiment,
            'Metal': list_metal,
            'Measurement': list_measurement,
            'Temperature': list_temperature,
            'Fit Range': list_fit_range,
            'Energy': list_energy,
            'Component Name': list_component_name,
            'Component': list_component,
            'Parameter Name': list_parameter_name,
            'Parameter Value': list_parameter_value,
            'Parameter Error': list_parameter_error
        })
        return None
    
    def PCA(self, n_components: Union[None, str, float, int, list]=None, fit_range: Union[None, tuple[float, float], list[tuple[float, float]]]=None, seed: Union[None, int]=None):
        if isinstance(n_components, list):
            if len(n_components) != len(self.experiments):
                raise ValueError('Length of list must match number of experiments')
        else:
            n_components = [n_components] * len(self.experiments)
        
        experiment_list = []
        metal_list = []
        measurement_list = []
        n_components_list = []
        pca_mean_list = []
        explained_variance_list = []
        explained_variance_ratio_list = []
        cumulative_explained_variance_list = []
        energy_list = []
        component_list = []
        component_names_list = []
        component_number_list = []
        weights_list = []
        
        for experiment, n_components in zip(self.experiments, n_components):
            if fit_range:
                raise NotImplementedError('Fit range not implemented yet')
            
            measurements = self.data['Measurement'][self.data['Experiment'] == experiment].unique()
            data = self.data['mu_norm'][self.data['Experiment'] == experiment].to_numpy().reshape(len(measurements), -1)
            
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(data)
            pca_weights = pca.transform(data)
            
            # Store PCA results
            for i, component in enumerate(pca.components_):
                for measurement in measurements:
                    experiment_list.append(experiment)
                    metal_list.append(self.data['Metal'][self.data['Experiment'] == experiment].values[0])
                    measurement_list.append(measurement)
                    n_components_list.append(pca.n_components_)
                    pca_mean_list.append(pca.mean_)
                    explained_variance_list.append(pca.explained_variance_[i])
                    explained_variance_ratio_list.append(pca.explained_variance_ratio_[i])
                    cumulative_explained_variance_list.append(pca.explained_variance_ratio_[:i+1].sum())
                    energy_list.append(self.data['Energy'][self.data['Experiment'] == experiment].to_numpy())
                    component_list.append(component)
                    component_names_list.append(f'PC {i+1}')
                    component_number_list.append(i+1)
                    weights_list.append(pca_weights[measurement-1,i])
                
        self.PCA_result = pd.DataFrame({
            'Experiment': experiment_list,
            'Metal': metal_list,
            'Measurement': measurement_list,
            'n_components': n_components_list,
            'PCA Mean': pca_mean_list,
            'Explained Variance': explained_variance_list,
            'Explained Variance Ratio': explained_variance_ratio_list,
            'Cumulative Explained Variance': cumulative_explained_variance_list,
            'Energy': energy_list,
            'Component': component_list,
            'Component Name': component_names_list,
            'Component Number': component_number_list,
            'Weight': weights_list
        })       
        return None
    
    def NMF(self, n_components: Union[None, str, float, int, list]=None, change_cutoff: float=0.25, fit_range: Union[None, tuple[float, float], list[tuple[float, float]]]=None, seed: Union[None, int]=None):
        if isinstance(n_components, list):
            if len(n_components) != len(self.experiments):
                raise ValueError('Length of list must match number of experiments')
        elif n_components is None:
            n_components = self._determine_NMF_components(change_cutoff=change_cutoff, fit_range=fit_range)
        else:
            n_components = [n_components] * len(self.experiments)
            
        experiment_list = []
        metal_list = []
        measurement_list = []
        n_components_list = []
        energy_list = []
        component_list = []
        component_names_list = []
        component_number_list = []
        weights_list = []
        
        for experiment, n_components in zip(self.experiments, n_components):
            if fit_range:
                raise NotImplementedError('Fit range not implemented yet')
            
            measurements = self.data['Measurement'][self.data['Experiment'] == experiment].unique()
            data = self.data['mu_norm'][self.data['Experiment'] == experiment].to_numpy().reshape(len(measurements), -1)
            # Remove negative values
            data -= data.min()
            
            nmf = NMF(n_components=n_components, random_state=seed, init='nndsvda')
            nmf.fit(data)
            nmf_weights = nmf.transform(data)
            
            # Store NMF results
            for i, component in enumerate(nmf.components_):
                for measurement in measurements:
                    experiment_list.append(experiment)
                    metal_list.append(self.data['Metal'][self.data['Experiment'] == experiment].values[0])
                    measurement_list.append(measurement)
                    n_components_list.append(nmf.n_components_)
                    energy_list.append(self.data['Energy'][self.data['Experiment'] == experiment].to_numpy())
                    component_list.append(component)
                    component_names_list.append(f'Component {i+1}')
                    component_number_list.append(i+1)
                    weights_list.append(nmf_weights[measurement-1,i])
        
        self.NMF_result = pd.DataFrame({
            'Experiment': experiment_list,
            'Metal': metal_list,
            'Measurement': measurement_list,
            'n_components': n_components_list,
            'Energy': energy_list,
            'Component': component_list,
            'Component Name': component_names_list,
            'Component Number': component_number_list,
            'Weight': weights_list
        })
        
        return None
    
    def _determine_NMF_components(self, change_cutoff: int, fit_range: Union[None, tuple[float, float], list[tuple[float, float]]]=None):
        
        experiment_list = []
        n_components_list = []
        reconstruction_error_list = []
        
        for experiment in self.experiments:
            if fit_range:
                raise NotImplementedError('Fit range not implemented yet')
            
            measurements = self.data['Measurement'][self.data['Experiment'] == experiment].unique()
            data = np.clip(self.data['mu_norm'][self.data['Experiment'] == experiment].to_numpy().reshape(len(measurements), -1), a_min=0, a_max=None)
            
            for n_components in range(len(measurements)):
                nmf = NMF(n_components=n_components + 1)
                nmf.fit(data)
                
                # Log reconstruction error
                experiment_list.append(experiment)
                n_components_list.append(n_components + 1)
                reconstruction_error_list.append(nmf.reconstruction_err_)
        
        NMF_component_results = pd.DataFrame({
            'Experiment': experiment_list,
            'n_components': n_components_list,
            'Reconstruction Error': reconstruction_error_list
        })
        
        NMF_component_results['Absolute Change'] = NMF_component_results.groupby('Experiment')['Reconstruction Error'].diff().abs()
        
        nmf_k = []
        for experiment in self.experiments:
            _nmf = NMF_component_results[NMF_component_results['Experiment'] == experiment]
            _derivative = np.absolute(np.gradient(_nmf['Reconstruction Error'], _nmf['n_components']))
            k = _nmf['n_components'][_derivative > np.abs(change_cutoff)].max()
            nmf_k.append(k)

        return nmf_k
        
    
    # Data export functions
    
    def to_csv(self, filename: str, directory: Union[None, str]=None, columns: Union[None, list[str]]=None):
        if directory is None:
            directory = self.save_directory
        self.data.to_csv(directory + 'data/' + filename + '.csv', index=False, columns=columns)
        return None
    
    def to_athena(self, filename: str, directory: str='./', columns: Union[None, list[str]]=None):
        # with open(directory + filename + '.nor', 'w') as file:
        #     file.write('# Exported from autoXAS')
        raise NotImplementedError('Athena export not implemented yet')
        return None
    
    # Plotting functions
    
    def plot_data(self, experiment: Union[str, int]=0, standards: Union[None, list[str]]=None, save: bool=False, filename: str='data', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
            
        n_measurements = int(self.data['Measurement'][self.data['Experiment'] == experiment].max())
        experiment_filter = (self.data['Experiment'] == experiment)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment
            fig = px.line(
                data_frame=self.data[experiment_filter],
                x='Energy',
                y='mu_norm',
                color='Measurement',
                color_discrete_sequence=px.colors.sample_colorscale('viridis', samplepoints=n_measurements),
            )
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            if standards:
                raise NotImplementedError('Standards not implemented yet')
            
            # Specify title text
            if show_title:
                title_text = f'<b>Normalized data<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title='<b>Normalized [a.u.]</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
                
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
                
        else:
            # Create figure object and set the figure size
            plt.figure(figsize=(10,8))
            
            # Plot all measurements of specified experiment
            sns.lineplot(
                data=self.data[experiment_filter], 
                x='Energy', 
                y='mu_norm', 
                hue='Measurement', 
                palette='viridis',
            )
            # Set limits of x-axis to match the edge measurements
            plt.xlim(
                (np.amin(self.data['Energy'][experiment_filter]), 
                np.amax(self.data['Energy'][experiment_filter]))
            )
            # Specify text and formatting of axis labels
            plt.xlabel(
                f'Energy [{self.energy_unit}]', 
                fontsize=14, 
                fontweight='bold'
            )
            plt.ylabel(
                'Normalized', 
                fontsize=14, 
                fontweight='bold'
            )
            # Specify placement, formatting and title of the legend
            plt.legend(
                loc='center left', 
                bbox_to_anchor=(1,0.5),
                title='Measurement', 
                fontsize=12, 
                title_fontsize=13, 
                ncol=1,
            )
            # Enforce matplotlibs tight layout
            plt.tight_layout()
            if save:
                plt.savefig(directory + 'figures/' + filename + format)
            
            if show:
                plt.show()
        return None
    
    def plot_temperature_curves(self, save: bool=False, filename: str='temperature_curve', format: str='.png', directory: Union[None, str]=None, show: bool=True, show_title: bool=True):
        if save and directory is None:
            directory = self.save_directory
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = '%{y:.1f} +/- %{customdata[0]:.1f} ' + self.temperature_unit
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = line(
                data_frame=self.data,
                x='Measurement',
                y='Temperature',
                error_y='Temperature (std)',
                error_y_mode='band',
                custom_data=['Temperature (std)'],
                color='Experiment',
                color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
            )
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Temperature curves</b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Temperature [{self.temperature_unit}]</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
    
    def plot_waterfall(self, experiment: Union[str, int]=0, y_axis: str='Measurement', vmin: Union[None, float]=None, vmax: Union[None, float]=None, save: bool=False, filename: str='waterfall', format: str='.png', directory: Union[None, str]=None, show: bool=True, show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
        
        experiment_filter = (self.data['Experiment'] == experiment)
        
        heatmap_data = self.data[experiment_filter].pivot(index=y_axis, columns='Energy', values='mu_norm')
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.2f'
            
            # Plot the measurements of the selected experiment
            fig = px.imshow(
                img = heatmap_data,
                zmin=vmin,
                zmax=vmax,
                origin='lower',
                color_continuous_scale='viridis',
                aspect='auto',
                labels=dict(color='<b>Normalized [a.u.]</b>'),
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b><i>In situ</i> overview<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title=f'<b>{y_axis}</b>',
                font=dict(
                    size=14,
                ),
                hovermode='closest',
                coloraxis=dict(
                    colorbar=dict(
                        titleside='right',
                    ),
                ),
            )
            
            hovertemplate = f'Measurement: %{{y}}<br>Energy: %{{x:{x_formatting}}} {self.energy_unit}<br>Normalized: %{{z:.2f}}<extra></extra>'
            fig.update_traces(hovertemplate=hovertemplate)
            
            # Add and format spikes
            fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
    
    def plot_change(self, experiment: Union[str, int]=0, reference_measurement: int=1, y_axis: str='Measurement', vmin: Union[None, float]=None, vmax: Union[None, float]=None, save: bool=False, filename: str='change', format: str='.png', directory: Union[None, str]=None, show: bool=True, show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
        
        experiment_filter = (self.data['Experiment'] == experiment)
        experiment_data = self.data[experiment_filter]
        reference = experiment_data['mu_norm'][(self.data['Measurement'] == reference_measurement)].to_numpy()
        difference_from_reference = experiment_data['mu_norm'].to_numpy().reshape((-1, reference.shape[0])) - reference
        experiment_data['Difference from reference'] = difference_from_reference.reshape((-1))
        
        heatmap_data = experiment_data.pivot(index=y_axis, columns='Energy', values='Difference from reference')
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.2f'
            
            # Plot the measurements of the selected experiment
            fig = px.imshow(
                img = heatmap_data,
                zmin=vmin,
                zmax=vmax,
                origin='lower',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0.,
                aspect='auto',
                labels=dict(color='<b>\u0394 Normalized [a.u.]</b>'),
            )
            
            fig.add_hline(
                y=experiment_data[y_axis].unique()[reference_measurement-1],
                line_color='black',
                line_width=2,
                annotation_text=f'<b>Reference (Measurement {reference_measurement})</b>',
                annotation_position='top left',
                annotation_font=dict(
                    color='black',
                    size=11,
                ),
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b><i>In situ</i> changes<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title=f'<b>{y_axis}</b>',
                font=dict(
                    size=14,
                ),
                hovermode='closest',
                coloraxis=dict(
                    colorbar=dict(
                        titleside='right',
                    ),
                ),
            )
            
            hovertemplate = f'Measurement: %{{y}}<br>Energy: %{{x:{x_formatting}}} {self.energy_unit}<br>\u0394 Normalized: %{{z:.2f}}<extra></extra>'
            fig.update_traces(hovertemplate=hovertemplate)
            
            # Add and format spikes
            fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
    
    def plot_LCA(self, experiment: Union[str, int]=0, save: bool=False, filename: str='LCA', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
            
        experiment_filter = (self.LCA_result['Experiment'] == experiment)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}} +/- %{{customdata[0]:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = line(
                data_frame=self.LCA_result[experiment_filter],
                x='Measurement',
                y='Parameter Value',
                error_y='Parameter Error',
                error_y_mode='band',
                custom_data=['Parameter Error'],
                color='Parameter Name',
                color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
                labels={'Parameter Name': 'Component'},
            )

            for trace in fig["data"]:
                if trace["name"] != None:
                    trace["name"] = f"Component {trace['name'][1:]}"
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Linear Combination Analysis<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None
    
    def plot_LCA_frame(self, experiment: Union[str, int]=0, measurement: int=1, save: bool=False, filename: str='LCA_frame', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory

        data_filter = (self.LCA_result['Experiment'] == experiment) & (self.LCA_result['Measurement'] == measurement)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot reference measurement
            reference = self.LCA_result['Component'][(self.LCA_result['Component Name'] == f'(Ref) Measurement {measurement}') & data_filter].values[0]
            
            fig = go.Figure(go.Scatter(
                x=self.LCA_result['Energy'][(self.LCA_result['Component Name'] == f'(Ref) Measurement {measurement}') & data_filter].values[0],
                y=reference,
                name='Data',
                mode='lines',
                line=dict(
                    color='black',
                    ),
            ))
            
            linear_combination = np.zeros_like(reference)
            # Plot components
            for i, component in enumerate(self.LCA_result['Component Name'][data_filter].unique()):
                if '(Ref)' in component:
                    continue
                fig.add_trace(go.Scatter(
                    x=self.LCA_result['Energy'][(self.LCA_result['Component Name'] == component) & data_filter].values[0],
                    y=self.LCA_result['Component'][(self.LCA_result['Component Name'] == component) & data_filter].values[0] * self.LCA_result['Parameter Value'][(self.LCA_result['Component Name'] == component) & data_filter].values[0],
                    name=f"Component {self.LCA_result['Parameter Name'][(self.LCA_result['Component Name'] == component) & data_filter].values[0][1:]}",
                    mode='lines',
                    line=dict(
                        color=sns.color_palette('colorblind').as_hex()[i],
                        ),
                ))
                
                linear_combination += self.LCA_result['Component'][(self.LCA_result['Component Name'] == component) & data_filter].values[0] * self.LCA_result['Parameter Value'][(self.LCA_result['Component Name'] == component) & data_filter].values[0]
            
            # Plot linear combination
            fig.add_trace(go.Scatter(
                x=self.LCA_result['Energy'][(self.LCA_result['Component Name'] == f'(Ref) Measurement {measurement}') & data_filter].values[0],
                y=linear_combination,
                name='Linear combination',
                mode='lines',
                line=dict(
                    color='magenta',
                    ),
            ))
            
            # Plot residuals
            fig.add_trace(go.Scatter(
                x=self.LCA_result['Energy'][(self.LCA_result['Component Name'] == f'(Ref) Measurement {measurement}') & data_filter].values[0],
                y=reference - linear_combination,
                name='Residual',
                mode='lines',
                line=dict(
                    color='red',
                    ),
            ))
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Linear Combination Analysis<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title='<b>Normalized [a.u.]</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
                
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
        
    def plot_LCA_comparison(self, component: int=2, save: bool=False, filename: str='LCA_comparison', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if save and directory is None:
            directory = self.save_directory
            
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}} +/- %{{customdata[0]:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = line(
                data_frame=self.LCA_result[self.LCA_result['Parameter Name'] == f'w{component}'],
                x='Measurement',
                y='Parameter Value',
                error_y='Parameter Error',
                error_y_mode='band',
                custom_data=['Parameter Error'],
                color='Metal',
                color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
            )
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>LCA Transition Comparison<br><sup><i>Component {component}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
    
    def plot_PCA(self, experiment: Union[str, int]=0, save: bool=False, filename: str='PCA', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
            
        experiment_filter = (self.PCA_result['Experiment'] == experiment)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = px.line(
                data_frame=self.PCA_result[experiment_filter],
                x='Measurement',
                y='Weight',
                color='Component Name',
                color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
                labels={'Parameter Name': 'Component'},
            )
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Principal Component Analysis<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        return None
    
    def plot_PCA_frame(self, experiment: Union[str, int]=0, measurement: int=1, save: bool=False, filename: str='PCA_frame', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory

        data_filter = (self.PCA_result['Experiment'] == experiment) & (self.PCA_result['Measurement'] == measurement)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot reference measurement
            reference = self.data['mu_norm'][(self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)].to_numpy()
            
            fig = go.Figure(go.Scatter(
                x=self.PCA_result['Energy'][(self.PCA_result['Component Name'] == f'PC 1') & data_filter].values[0],
                y=reference,
                name='Data',
                mode='lines',
                line=dict(
                    color='black',
                    ),
            ))
            
            # Plot PCA mean
            pca_mean = self.PCA_result['PCA Mean'][data_filter].values[0] 
            
            fig.add_trace(go.Scatter(
                x=self.PCA_result['Energy'][(self.PCA_result['Component Name'] == f'PC 1') & data_filter].values[0],
                y=pca_mean,
                name='PCA mean',
                mode='lines',
                line=dict(
                    color='grey',
                    ),
            ))
            
            pca_reconstruction = np.zeros_like(reference)
            
            # Plot components
            for i, component in enumerate(self.PCA_result['Component Name'][data_filter].unique()):
                fig.add_trace(go.Scatter(
                    x=self.PCA_result['Energy'][(self.PCA_result['Component Name'] == component) & data_filter].values[0],
                    y=self.PCA_result['Component'][(self.PCA_result['Component Name'] == component) & data_filter].values[0] * self.PCA_result['Weight'][(self.PCA_result['Component Name'] == component) & data_filter].values[0],
                    name=component,
                    mode='lines',
                    line=dict(
                        color=sns.color_palette('colorblind').as_hex()[i],
                        ),
                ))

                pca_reconstruction += self.PCA_result['Component'][(self.PCA_result['Component Name'] == component) & data_filter].values[0] * self.PCA_result['Weight'][(self.PCA_result['Component Name'] == component) & data_filter].values[0]
            
            pca_reconstruction += pca_mean
            
            # Plot PCA reconstruction
            fig.add_trace(go.Scatter(
                x=self.PCA_result['Energy'][(self.PCA_result['Component Name'] == f'PC 1') & data_filter].values[0],
                y=pca_reconstruction,
                name='PCA reconstruction',
                mode='lines',
                line=dict(
                    color='magenta',
                    ),
            ))
            
            # Plot residual
            fig.add_trace(go.Scatter(
                x=self.PCA_result['Energy'][(self.PCA_result['Component Name'] == f'PC 1') & data_filter].values[0],
                y=reference - pca_reconstruction,
                name='Residual',
                mode='lines',
                line=dict(
                    color='red',
                    ),
            ))
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Principal Component Analysis<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title='<b>Normalized [a.u.]</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
                
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
    
    def plot_PCA_comparison(self, component: Union[int, list[int]]=1, save: bool=False, filename: str='PCA_comparison', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if save and directory is None:
            directory = self.save_directory
        
        if isinstance(component, list):
            if len(component) != len(self.experiments):
                raise ValueError('Length of list must match number of experiments')
        else:
            component = [component]*len(self.experiments)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = go.Figure()
            
            for i, experiment in enumerate(self.experiments):
                data_filter = (self.PCA_result['Experiment'] == experiment) & (self.PCA_result['Component Number'] == component[i])
                
                fig.add_trace(go.Scatter(
                    x=self.PCA_result['Measurement'][data_filter],
                    y=self.PCA_result['Weight'][data_filter],
                    name=self.PCA_result['Metal'][data_filter].values[0] + f' (PC {component[i]})',
                    mode='lines',
                    line=dict(
                        color=sns.color_palette('colorblind').as_hex()[i],
                        ),
                ))
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>PCA Transition Comparison</b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None
    
    def plot_PCA_explained_variance(self, plot_type: str='cumulative', variance_threshold: Union[None, float]=None, fit_range: Union[None, tuple[float, float], list[tuple[float, float]]]=None, seed: Union[None, int]=None, save: bool=False, filename: str='PCA_explained_variance', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2%', show_title: bool=True):
        
        if save and directory is None:
            directory = self.save_directory
            
        if plot_type not in ['ratio', 'cumulative']:
            raise ValueError('Invalid plot type. Choose between "ratio" and "cumulative"')
        
        experiment_list = []
        metal_list = []
        component_number_list = []
        component_name_list = []
        explained_variance_list = []
        explained_variance_ratio_list = []
        cumulative_explained_variance_list = []
        
        for experiment in self.experiments:
            if fit_range:
                raise NotImplementedError('Fit range not implemented yet')
            
            measurements = self.data['Measurement'][self.data['Experiment'] == experiment].unique()
            data = self.data['mu_norm'][self.data['Experiment'] == experiment].to_numpy().reshape(len(measurements), -1)
            
            pca = PCA(random_state=seed)
            pca.fit(data)
            
            # Store PCA results
            for i in range(pca.n_components_ + 1):
                experiment_list.append(experiment)
                metal_list.append(self.data['Metal'][self.data['Experiment'] == experiment].values[0])
                component_number_list.append(i)
                component_name_list.append(f'PC {i}')
                if i == 0:
                    explained_variance_list.append(0)
                    explained_variance_ratio_list.append(0)
                    cumulative_explained_variance_list.append(0)
                else:
                    explained_variance_list.append(pca.explained_variance_[i-1])
                    explained_variance_ratio_list.append(pca.explained_variance_ratio_[i-1])
                    cumulative_explained_variance_list.append(pca.explained_variance_ratio_[:i].sum())
        
        pca_results = pd.DataFrame({
            'Experiment': experiment_list,
            'Metal': metal_list,
            'Component Number': component_number_list,
            'Component Name': component_name_list,
            'Explained Variance': explained_variance_list,
            'Explained Variance Ratio': explained_variance_ratio_list,
            'Cumulative Explained Variance': cumulative_explained_variance_list,
        })
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            if plot_type == 'ratio':
                fig = px.bar(
                    data_frame=pca_results[pca_results['Component Number'] != 0],
                    x='Component Name',
                    y='Explained Variance Ratio',
                    color='Metal',
                    color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
                    barmode='group',
                )
                
                # Specify axis titles
                xaxis_title = '<b>Principal Components</b>'
                yaxis_title = '<b>Explained Variance</b>'
                
                # Specify title text
                if show_title:
                    title_text = f'<b>Explained Variance Ratio</b>'
                else:
                    title_text = ''
                
                # Change bar formatting
                fig.update_traces(
                    marker=dict(
                        line=dict(
                            width=1,
                        ),
                    ),
                    xhoverformat=x_formatting,
                    hovertemplate=hovertemplate,
                )
                
            elif plot_type == 'cumulative':
                fig = px.line(
                    data_frame=pca_results,
                    x='Component Number',
                    y='Cumulative Explained Variance',
                    color='Metal',
                    color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
                )
                
                # Specify axis titles
                xaxis_title = '<b>Number of Principal Components</b>'
                yaxis_title = '<b>Explained Variance</b>'
                
                # Specify title text
                if show_title:
                    title_text = f'<b>Cumulative Explained Variance</b>'
                else:
                    title_text = ''
                
                if variance_threshold:
                    fig.add_hline(
                        y=variance_threshold,
                        line_color='black',
                        line_width=2,
                        line_dash='dash',
                        annotation_text=f'<b>{variance_threshold:.0%}</b>',
                        annotation_position='bottom right',
                        annotation_font=dict(
                            color='black',
                            size=12,
                        ),
                    )                
                
                # Change line formatting
                fig.update_traces(
                    line=dict(
                        width=2,
                    ),
                    xhoverformat=x_formatting,
                    hovertemplate=hovertemplate,
                )
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
                yaxis=dict(
                    tickformat='.0%',
                ),
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
            
            if show:
                fig.show()
            
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None
    
    def plot_NMF(self, experiment: Union[str, int]=0, save: bool=False, filename: str='NMF', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')

        if save and directory is None:
            directory = self.save_directory
        
        experiment_filter = (self.NMF_result['Experiment'] == experiment)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = px.line(
                data_frame=self.NMF_result[experiment_filter],
                x='Measurement',
                y='Weight',
                color='Component Name',
                color_discrete_sequence=sns.color_palette('colorblind').as_hex(),
                labels={'Parameter Name': 'Component'},
            )
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Non-negative Matrix Factorization<br><sup><i>{experiment}</i></sup></b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
        
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None
    
    def plot_NMF_frame(self, experiment: Union[str, int]=0, measurement: int=1, save: bool=False, filename: str='NMF_frame', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError('Invalid experiment name')
        
        if save and directory is None:
            directory = self.save_directory
        
        data_filter = (self.NMF_result['Experiment'] == experiment) & (self.NMF_result['Measurement'] == measurement)
        
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot reference measurement
            reference = self.data['mu_norm'][(self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)].to_numpy()
            
            fig = go.Figure(go.Scatter(
                x=self.NMF_result['Energy'][(self.NMF_result['Component Name'] == f'Component 1') & data_filter].values[0],
                y=reference,
                name='Data',
                mode='lines',
                line=dict(
                    color='black',
                    ),
            ))
            
            nmf_reconstruction = np.zeros_like(reference)
            
            # Plot components
            for i, component in enumerate(self.NMF_result['Component Name'][data_filter].unique()):
                fig.add_trace(go.Scatter(
                    x=self.NMF_result['Energy'][(self.NMF_result['Component Name'] == component) & data_filter].values[0],
                    y=self.NMF_result['Component'][(self.NMF_result['Component Name'] == component) & data_filter].values[0] * self.NMF_result['Weight'][(self.NMF_result['Component Name'] == component) & data_filter].values[0],
                    name=component,
                    mode='lines',
                    line=dict(
                        color=sns.color_palette('colorblind').as_hex()[i],
                        ),
                ))

                nmf_reconstruction += self.NMF_result['Component'][(self.NMF_result['Component Name'] == component) & data_filter].values[0] * self.NMF_result['Weight'][(self.NMF_result['Component Name'] == component) & data_filter].values[0]
            
            # Plot NMF reconstruction
            fig.add_trace(go.Scatter(
                x=self.NMF_result['Energy'][(self.NMF_result['Component Name'] == f'Component 1') & data_filter].values[0],
                y=nmf_reconstruction,
                name='NMF reconstruction',
                mode='lines',
                line=dict(
                    color='magenta',
                    ),
            ))

            # Plot residual
            fig.add_trace(go.Scatter(
                x=self.NMF_result['Energy'][(self.NMF_result['Component Name'] == f'Component 1') & data_filter].values[0],
                y=reference - nmf_reconstruction,
                name='Residual',
                mode='lines',
                line=dict(
                    color='red',
                    ),
            ))
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>Non-negative Matrix Factorization<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>'
            else:
                title_text = ''
                
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title=f'<b>Energy [{self.energy_unit}]</b>',
                yaxis_title='<b>Normalized [a.u.]</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
        
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None
    
    def plot_NMF_comparison(self, component: Union[int, list[int]]=1, save: bool=False, filename: str='NMF_comparison', format: str='.png', directory: Union[None, str]=None, show: bool=True, hover_format: str='.2f', show_title: bool=True):
        if save and directory is None:
            directory = self.save_directory
            
        if isinstance(component, list):
            if len(component) != len(self.experiments):
                raise ValueError('Length of list must match number of experiments')
        else:
            component = [component]*len(self.experiments)
            
        if self.interactive:
            # Formatting for hover text
            x_formatting = '.0f'
            hovertemplate = f'%{{y:{hover_format}}}'
            hovermode='x unified'
            
            # Plot the measurements of the selected experiment/edge
            fig = go.Figure()
            
            for i, experiment in enumerate(self.experiments):
                data_filter = (self.NMF_result['Experiment'] == experiment) & (self.NMF_result['Component Number'] == component[i])
                
                fig.add_trace(go.Scatter(
                    x=self.NMF_result['Measurement'][data_filter],
                    y=self.NMF_result['Weight'][data_filter],
                    name=self.NMF_result['Metal'][data_filter].values[0] + f' (Component {component[i]})',
                    mode='lines',
                    line=dict(
                        color=sns.color_palette('colorblind').as_hex()[i],
                        ),
                ))
            
            # Change line formatting
            fig.update_traces(
                line=dict(
                    width=2,
                ),
                xhoverformat=x_formatting,
                hovertemplate=hovertemplate,
            )
            
            # Specify title text
            if show_title:
                title_text = f'<b>NMF Transition Comparison</b>'
            else:
                title_text = ''
            
            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x = 0.5,
                xaxis_title='<b>Measurement</b>',
                yaxis_title=f'<b>Weight</b>',
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )
            
            if save:
                fig.write_image(directory + 'figures/' + filename + format)
                
            if show:
                fig.show()
                
        else:
            raise NotImplementedError('Matplotlib plot not implemented yet')
        
        return None