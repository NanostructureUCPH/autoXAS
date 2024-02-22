# %% Imports

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm.auto import tqdm
from larch import Group
from larch.xafs import pre_edge
from typing import Union

# %% autoXAS class

class autoXAS():
    def __init__(self) -> None:
        self.data_directory = None
        self.data_type = '.dat'
        self.data = None
        self.raw_data = None
        self.experiments = None
        self.save_directory = './'
        self.energy_column = None
        self.I0_columns = None
        self.I1_columns = None
        self.temperature_column = None
        self.metals = None
        self.edge_correction = False
        self.xas_mode = 'Flourescence'
        self.energy_unit = 'eV'
        self.energy_column_unitConversion = 1
    
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
                    self.data = pd.concat([self.data, data])
        elif self.data_type == '.h5':
            raise NotImplementedError('HDF5 file reading not implemented yet')
        return None
    
    def _edge_correction(self):
        raise NotImplementedError('Edge correction not implemented yet')
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
                    energy_avg = np.zeros_like(df_avg['Energy'], dtype=np.float64)
                    i0_avg = np.zeros_like(df_avg['I0'], dtype=np.float64)
                    i1_avg = np.zeros_like(df_avg['I1'], dtype=np.float64)
                    mu_avg = np.zeros_like(df_avg['mu'], dtype=np.float64)
                    temperature_avg = np.zeros_like(df_avg['Temperature'], dtype=np.float64)
                    
                    first = False
                
                energy_avg += self.data['Energy'][measurement_filter].to_numpy()
                i0_avg += self.data['I0'][measurement_filter].to_numpy()
                i1_avg += self.data['I1'][measurement_filter].to_numpy()
                mu_avg += self.data['mu'][measurement_filter].to_numpy()
                temperature_avg += self.data['Temperature'][measurement_filter].to_numpy()
            
            n_measurements = len(measurements)
            df_avg['Energy'] = energy_avg / n_measurements
            df_avg['I0'] = i0_avg / n_measurements
            df_avg['I1'] = i1_avg / n_measurements
            df_avg['mu'] = mu_avg / n_measurements
            df_avg['Temperature'] = temperature_avg / n_measurements
            
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
                        temperature_avg = np.zeros_like(df_avg['Temperature'], dtype=np.float64)
                        
                    energy_avg += self.data['Energy'][measurement_filter].to_numpy()
                    i0_avg += self.data['I0'][measurement_filter].to_numpy()
                    i1_avg += self.data['I1'][measurement_filter].to_numpy()
                    mu_avg += self.data['mu'][measurement_filter].to_numpy()
                    temperature_avg += self.data['Temperature'][measurement_filter].to_numpy()
                    
                n_measurements = len(measurements_to_average_temp)
                df_avg['Energy'] = energy_avg / n_measurements
                df_avg['I0'] = i0_avg / n_measurements
                df_avg['I1'] = i1_avg / n_measurements
                df_avg['mu'] = mu_avg / n_measurements
                df_avg['Temperature'] = temperature_avg / n_measurements
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
    
    def load_data(self, average: bool=False, average_type: str='standard', measurements_to_average: Union[str, list[int], np.ndarray, range]='all', n_periods: Union[None, int]=None, period: Union[None, int]=None):	
        self._read_data()
        self.experiments = list(self.data['Experiment'].unique())
        self.metals = list(self.data['Metal'].unique())
        if average:
            if average_type == 'standard':
                self._average_data(measurements_to_average=measurements_to_average)
            elif average_type == 'periodic':
                self._average_data_periodic(period=period, n_periods=n_periods)
            else:
                raise ValueError('Invalid average_type. Must be "standard" or "periodic"')
        self._normalize_data()
        return None
    
    def LCA(self):
        raise NotImplementedError('LCA not implemented yet')
        return None
    
    def PCA(self):
        raise NotImplementedError('PCA not implemented yet')
        return None
    
    def NMF(self):
        raise NotImplementedError('NMF not implemented yet')
        return None
    
    def to_csv(self, filename: str, directory: str='./', columns: Union[None, list[str]]=None):
        self.data.to_csv(directory + filename + '.csv', index=False, columns=columns)
        return None
    
    def to_athena(self, filename: str, directory: str='./', columns: Union[None, list[str]]=None):
        # with open(directory + filename + '.nor', 'w') as file:
        #     file.write('# Exported from autoXAS')
        raise NotImplementedError('Athena export not implemented yet')
        return None
    
    def plot_data(self):
        raise NotImplementedError('Plotting not implemented yet')
        return None
    
    def plot_temperature_curve(self):
        raise NotImplementedError('Plotting not implemented yet')
        return None
    
    def plot_LCA(self):
        raise NotImplementedError('Plotting not implemented yet')
        return None
    
    def plot_PCA(self):
        raise NotImplementedError('Plotting not implemented yet')
        return None
    
    def plot_NMF(self):
        raise NotImplementedError('Plotting not implemented yet')
        return None