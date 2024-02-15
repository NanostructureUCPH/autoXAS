# %% Imports

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm.auto import tqdm
from larch import Group
from larch.xafs import pre_edge

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
            for file in tqdm(data_files, desc='Reading data files'):
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
                for energy_step in data['Energy'].diff():
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
    
    def _average_data(self):
        pass
    
    def _average_data_periodic(self):
        pass
    
    def _normalize_data(self):
        if self.data is None:
            raise ValueError('No data to normalize')
        
        self.data['mu_norm'] = 0
        self.data['pre_edge'] = 0
        self.data['post_edge'] = 0
        
        for experiment in tqdm(self.experiments, desc='Normalizing data'):
            print(experiment)
            experiment_filter = (self.data['Experiment'] == experiment)
            if self.edge_correction:
                raise NotImplementedError('Edge correction not implemented yet')
            for measurement in self.data['Measurement'][experiment_filter].unique():
                print(measurement)
                measurement_filter = (self.data['Experiment'] == experiment) & (self.data['Measurement'] == measurement)
                dummy_group = Group(name='dummy')
                
                self.data['mu_norm'][measurement_filter] = self.data['mu'][measurement_filter] - np.amin(self.data['mu'][measurement_filter])
                print(self.data['mu_norm'][measurement_filter])
                print(self.data['Energy'][measurement_filter])
                try:
                    pre_edge(self.data['Energy'][measurement_filter], self.data['mu_norm'][measurement_filter], group=dummy_group)
                    self.data['pre_edge'][measurement_filter] = dummy_group.pre_edge
                    self.data['post_edge'][measurement_filter] = dummy_group.post_edge
                    self.data['mu_norm'][measurement_filter] -= dummy_group.pre_edge
                    pre_edge(self.data['Energy'][measurement_filter], self.data['mu_norm'][measurement_filter], group=dummy_group)
                    self.data['mu_norm'][measurement_filter] /= dummy_group.post_edge
                except:
                    # self.data.drop(self.data[measurement_filter].index, inplace=True)
                    print(f'Error normalizing {experiment} measurement {measurement}. Measurement removed.')
        return None
    
    def load_data(self):
        self._read_data()
        self.experiments = list(self.data['Experiment'].unique())
        self.metals = list(self.data['Metal'].unique())
        self._normalize_data()
        return None
    