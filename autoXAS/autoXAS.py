# %% Imports


# Standard library imports
from pathlib import Path
from typing import Union

# Package imports
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
from lmfit import Parameters, minimize
from sklearn.decomposition import NMF, PCA
from tqdm.auto import tqdm

# %% Setup


pd.options.mode.chained_assignment = None  # default='warn'
sns.set_theme()
pio.renderers.default = "notebook"

# %% Other functions


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {"bar", "band", "bars", "bands", None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(
            f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}."
        )
    if error_y_mode in {"bar", "bars", None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {"band", "bands"}:
        if "error_y" not in kwargs:
            raise ValueError(
                f"If you provide argument 'error_y_mode' you must also provide 'error_y'."
            )
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg, val in kwargs.items() if arg != "error_y"})
        for data in figure_with_error_bars.data:
            x = list(data["x"])
            y_upper = list(data["y"] + data["error_y"]["array"])
            y_lower = list(
                data["y"] - data["error_y"]["array"]
                if data["error_y"]["arrayminus"] is None
                else data["y"] - data["error_y"]["arrayminus"]
            )
            color = (
                f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace(
                    "((", "("
                )
                .replace("),", ",")
                .replace(" ", "")
            )
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=data["legendgroup"],
                    xaxis=data["xaxis"],
                    yaxis=data["yaxis"],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data) / 2)):
            reordered_data.append(fig.data[i + int(len(fig.data) / 2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


# %% autoXAS class


class autoXAS:
    def __init__(self, metals=None, edges=None) -> None:
        self.data_directory = None
        self.data_type = ".dat"
        self.data = None
        self.raw_data = None
        self.standards_directory = None
        self.standards = None
        self.raw_standards = None
        self.experiments = None
        self.save_directory = "./"
        self.energy_column = None
        self.I0_columns = None
        self.I1_columns = None
        self.temperature_column = None
        self.metals = None
        self.xas_mode = "Flourescence"
        self.energy_unit = "eV"
        self.energy_column_unitConversion = 1
        self.temperature_unit = "K"
        self.interactive = False
        self.edge_correction_energies = {}
        if metals and edges:
            self._calculate_edge_shift(metals, edges)

    def save_config(self, config_name: str, save_directory: str = "./") -> None:
        """
        Save configuration file.

        Args:
            config_name (str): Name of the configuration file.
            save_directory (str, optional): Directory where the configuration file will be saved. Defaults to "./".

        Returns:
            None: Function does not return anything.
        """
        config = dict(
            data_directory=self.data_directory,
            standards_directory=self.standards_directory,
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
            save_directory=self.save_directory,
        )
        with open(save_directory + config_name, "w") as file:
            yaml.dump(config, file)

        return None

    def load_config(self, config_name: str, directory: str = "./") -> None:
        """
        Load configuration file.

        Args:
            config_name (str): Name of the configuration file.
            directory (str, optional): Directory where the configuration file is located. Defaults to "./".

        Returns:
            None: Function does not return anything.
        """
        with open(directory + config_name, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.data_directory = config["data_directory"]
        self.standards_directory = config["standards_directory"]
        self.data_type = config["data_type"]
        self.energy_column = config["energy_column"]
        self.I0_columns = config["I0_columns"]
        self.I1_columns = config["I1_columns"]
        self.temperature_column = config["temperature_column"]
        self.edge_correction = config["edge_correction"]
        self.xas_mode = config["xas_mode"]
        self.energy_unit = config["energy_unit"]
        self.energy_column_unitConversion = config["energy_column_unitConversion"]
        self.temperature_unit = config["temperature_unit"]
        self.save_directory = config["save_directory"]
        return None

    def _read_data(self, standards: bool = False) -> None:
        """
        Read data files and store in DataFrame.

        Args:
            standards (bool, optional): Whether to read standards. Defaults to False.

        Raises:
            ValueError: No data directory specified.
            NotImplementedError: HDF5 file reading not implemented yet.

        Returns:
            None: Function does not return anything.
        """
        if standards:
            if self.standards_directory is None:
                raise ValueError("No standards directory specified")
        else:
            if self.data_directory is None:
                raise ValueError("No data directory specified")

        if self.data_type == ".dat":
            if standards:
                data_files = list(Path(self.standards_directory).rglob("*.dat"))
            else:
                data_files = list(Path(self.data_directory).rglob("*.dat"))
            for file in tqdm(data_files, desc="Reading data files", leave=False):
                rows_to_skip = 0
                last_line = ""
                with open(file, "r") as f:
                    for line in f:
                        if line.startswith("\n") or line.startswith("#"):
                            last_line = line
                            rows_to_skip += 1
                        else:
                            columns = last_line.split()[1:]
                            break

                raw_data = pd.read_csv(
                    file,
                    sep="\s+",
                    header=None,
                    names=columns,
                    skiprows=rows_to_skip,
                    skip_blank_lines=True,
                    on_bad_lines="skip",
                    keep_default_na=False,
                )

                raw_data = raw_data.apply(
                    pd.to_numeric, errors="coerce"
                )  # , downcast='float')
                raw_data.dropna(inplace=True)

                data = pd.DataFrame()

                raw_data["File"] = file.name

                data["File"] = raw_data["File"]
                data["Experiment"] = file.stem
                for fragment in file.stem.split("_"):
                    if len(fragment) < 3 and fragment.isalpha():
                        data["Metal"] = fragment

                # Calculate energy
                data["Energy"] = (
                    raw_data[self.energy_column] * self.energy_column_unitConversion
                )

                # Calculate temperature
                if self.temperature_column is not None:
                    data["Temperature"] = raw_data[self.temperature_column]
                else:
                    data["Temperature"] = 0  # Placeholder for temperature
                data["Temperature (std)"] = 0  # Placeholder for temperature std

                # Calculate I0
                if isinstance(self.I0_columns, list):
                    data["I0"] = 0
                    for column in self.I0_columns:
                        data["I0"] += raw_data[column]
                elif isinstance(self.I0_columns, str):
                    data["I0"] = raw_data[self.I0_columns]
                # Calculate I1
                if isinstance(self.I1_columns, list):
                    data["I1"] = 0
                    for column in self.I1_columns:
                        data["I1"] += raw_data[column]
                elif isinstance(self.I1_columns, str):
                    data["I1"] = raw_data[self.I1_columns]

                # Calculate absorption coefficient
                if self.xas_mode == "Flourescence":
                    data["mu"] = data["I1"] / data["I0"]
                elif self.xas_mode == "Transmission":
                    data["mu"] = np.log(data["I0"] / data["I1"])

                # Remove data points with energy = 0
                data = data[data["Energy"] != 0]

                # Determine which measurement each data point belongs to
                measurement_number = 1
                measurement_number_values = []
                for energy_step in data["Energy"].diff().round(1):
                    if energy_step < 0:
                        measurement_number += 1
                    measurement_number_values.append(int(measurement_number))
                data["Measurement"] = measurement_number_values

                # Specify data types in specific columns
                data = data.astype(
                    {"Experiment": str, "Metal": str, "Measurement": int}
                )

                if standards:
                    if self.raw_standards is None:
                        self.raw_standards = raw_data
                    else:
                        self.raw_standards = pd.concat([self.raw_standards, raw_data])

                    if self.standards is None:
                        self.standards = data
                    else:
                        self.standards = pd.concat([self.standards, data]).reset_index(
                            drop=True
                        )
                else:
                    if self.raw_data is None:
                        self.raw_data = raw_data
                    else:
                        self.raw_data = pd.concat([self.raw_data, raw_data])

                    if self.data is None:
                        self.data = data
                    else:
                        self.data = pd.concat([self.data, data]).reset_index(drop=True)

        elif self.data_type == ".h5":
            raise NotImplementedError("HDF5 file reading not implemented yet")
        return None

    def _calculate_edge_shift(self, metals: list[str], edges: list[str]) -> None:
        """
        Calculate the shift in edge energy for each metal and edge pair.

        Args:
            metals (list[str]): List of metals.
            edges (list[str]): List of edges.

        Returns:
            None: Function does not return anything.
        """
        for metal, edge in zip(metals, edges):
            edge_energy_table = xray_edge(metal, edge, energy_only=True)

            measurement_filter = (self.data["Metal"] == metal) & (
                self.data["Experiment"] == 1
            )
            edge_energy_measured = find_e0(
                self.data["Energy"][measurement_filter],
                self.data["mu"][measurement_filter],
            )
            self.edge_correction_energies[metal] = (
                edge_energy_table - edge_energy_measured
            )
        return None

    def _energy_correction(self, standards: bool = False) -> None:
        """
        Correct energy range and align measured energy points for each experiment.

        Args:
            standards (bool, optional): Whether to correct standards. Defaults to False.

        Returns:
            None: Function does not return anything.
        """
        dataframe = self.standards if standards else self.data
        experiments = self.standard_experiments if standards else self.experiments

        for experiment in tqdm(experiments, desc="Energy correction", leave=False):
            experiment_filter = dataframe["Experiment"] == experiment
            n_measurements = dataframe["Measurement"][experiment_filter].max()
            # Correct for small variations in measured energy points
            energy = (
                dataframe["Energy"][experiment_filter]
                .to_numpy()
                .reshape(n_measurements, -1)
            )
            energy_correction = energy.mean(axis=0)
            # Correct for edge shift
            energy_correction += self.edge_correction_energies.get(
                dataframe["Metal"][experiment_filter].values[0], 0.0
            )
            # Estimate mu at corrected energy points using linear interpolation
            for measurement in range(1, n_measurements + 1):
                measurement_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"] == measurement
                )
                i0_interpolated = np.interp(
                    energy_correction,
                    dataframe["Energy"][measurement_filter],
                    dataframe["I0"][measurement_filter],
                )
                i1_interpolated = np.interp(
                    energy_correction,
                    dataframe["Energy"][measurement_filter],
                    dataframe["I1"][measurement_filter],
                )
                mu_interpolated = np.interp(
                    energy_correction,
                    dataframe["Energy"][measurement_filter],
                    dataframe["mu"][measurement_filter],
                )

                # Apply correction
                dataframe["Energy"][measurement_filter] = energy_correction
                dataframe["I0"][measurement_filter] = i0_interpolated
                dataframe["I1"][measurement_filter] = i1_interpolated
                dataframe["mu"][measurement_filter] = mu_interpolated

            if standards:
                self.standards = dataframe
            else:
                self.data = dataframe
        return None

    def _average_data(
        self,
        measurements_to_average: Union[str, list[int], np.ndarray, range] = "all",
        standards: bool = False,
    ) -> None:
        """
        Average data points for each experiment.

        Args:
            measurements_to_average (Union[str, list[int], np.ndarray, range], optional): Measurements to average. Defaults to "all".
            standards (bool, optional): Whether to average standards. Defaults to False.

        Returns:
            None: Function does not return anything.
        """
        dataframe = self.standards if standards else self.data
        experiments = self.standard_experiments if standards else self.experiments

        avg_measurements = []
        for experiment in tqdm(experiments, desc="Averaging data", leave=False):
            first = True
            experiment_filter = dataframe["Experiment"] == experiment

            if measurements_to_average == "all":
                measurements = dataframe["Measurement"][experiment_filter].unique()
            else:
                measurements = measurements_to_average

            for measurement in measurements:
                measurement_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"] == measurement
                )

                if first:
                    df_avg = dataframe[measurement_filter].copy()
                    i0_avg = np.zeros_like(df_avg["I0"], dtype=np.float64)
                    i1_avg = np.zeros_like(df_avg["I1"], dtype=np.float64)
                    mu_avg = np.zeros_like(df_avg["mu"], dtype=np.float64)

                    first = False

                i0_avg += dataframe["I0"][measurement_filter].to_numpy()
                i1_avg += dataframe["I1"][measurement_filter].to_numpy()
                mu_avg += dataframe["mu"][measurement_filter].to_numpy()

            n_measurements = len(measurements)
            df_avg["I0"] = i0_avg / n_measurements
            df_avg["I1"] = i1_avg / n_measurements
            df_avg["mu"] = mu_avg / n_measurements
            df_avg["Temperature"] = dataframe["Temperature"][measurement_filter].mean()
            df_avg["Temperature (std)"] = dataframe["Temperature"][
                measurement_filter
            ].std()

            avg_measurements.append(df_avg)

        if standards:
            self.standards = pd.concat(avg_measurements)
        else:
            self.data = pd.concat(avg_measurements)
        return None

    def _average_data_periodic(
        self,
        period: Union[None, int] = None,
        n_periods: Union[None, int] = None,
        standards: bool = False,
    ) -> None:
        """
        Average data points for each experiment using periodic grouping of measurements.

        Args:
            period (Union[None, int], optional): Number of measurements to group for averaging. Will determine the number of periods automatically. Defaults to None.
            n_periods (Union[None, int], optional): Number of periods to group for averaging. Will determine the number of measurements per period automatically. Defaults to None.

        Raises:
            Exception: Exactly 1 optional argument should be given.

        Returns:
            None: Function does not return anything.
        """

        dataframe = self.standards if standards else self.data
        experiments = self.standard_experiments if standards else self.experiments

        avg_measurements = []
        if (period and n_periods) or (not period and not n_periods):
            n_arguments = bool(period) + bool(n_periods)
            raise Exception(
                f"Exactly 1 optional argument should be given. {n_arguments} was given."
            )
        for experiment in tqdm(experiments, desc="Averaging data", leave=False):
            experiment_filter = dataframe["Experiment"] == experiment
            n_total_measurements = np.amax(dataframe["Measurement"][experiment_filter])
            if period:
                n_measurements_to_average = period
                new_n_measurements = int(np.ceil(n_total_measurements / period))
            elif n_periods:
                n_measurements_to_average = n_total_measurements // n_periods
                new_n_measurements = n_periods

            measurements_to_average = np.arange(n_measurements_to_average) + 1
            measurements_to_average_temp = measurements_to_average.copy()

            for measurement_number in range(new_n_measurements):
                if measurements_to_average_temp.any() >= n_total_measurements:
                    measurements_to_average_temp = np.array(
                        [
                            i
                            for i in measurements_to_average_temp
                            if i < n_total_measurements
                        ]
                    )

                for measurement in measurements_to_average_temp:
                    measurement_filter = (dataframe["Experiment"] == experiment) & (
                        dataframe["Measurement"] == measurement
                    )
                    if measurement == measurements_to_average_temp[0]:
                        df_avg = dataframe[measurement_filter].copy()
                        energy_avg = np.zeros_like(df_avg["Energy"], dtype=np.float64)
                        i0_avg = np.zeros_like(df_avg["I0"], dtype=np.float64)
                        i1_avg = np.zeros_like(df_avg["I1"], dtype=np.float64)
                        mu_avg = np.zeros_like(df_avg["mu"], dtype=np.float64)

                    energy_avg += dataframe["Energy"][measurement_filter].to_numpy()
                    i0_avg += dataframe["I0"][measurement_filter].to_numpy()
                    i1_avg += dataframe["I1"][measurement_filter].to_numpy()
                    mu_avg += dataframe["mu"][measurement_filter].to_numpy()

                n_measurements = len(measurements_to_average_temp)
                df_avg["Energy"] = energy_avg / n_measurements
                df_avg["I0"] = i0_avg / n_measurements
                df_avg["I1"] = i1_avg / n_measurements
                df_avg["mu"] = mu_avg / n_measurements
                df_avg["Temperature"] = dataframe["Temperature"][
                    measurement_filter
                ].mean()
                df_avg["Temperature (std)"] = dataframe["Temperature"][
                    measurement_filter
                ].std()
                df_avg["Measurement"] = measurement_number + 1

                measurements_to_average_temp += n_measurements_to_average

                avg_measurements.append(df_avg)

        if standards:
            self.standards = pd.concat(avg_measurements)
        else:
            self.data = pd.concat(avg_measurements)
        return None

    def _normalize_data(self, standards: bool = False) -> None:
        """
        Normalize data by subtracting the pre-edge fit and dividing by the post-edge fit.

        Pre- and post-edge fits are done using the `larch.xafs.pre_edge` function.

        Args:
            standards (bool, optional): Whether to normalize standards. Defaults to False.

        Raises:
            ValueError: No data to normalize.

        Returns:
            None: Function does not return anything.
        """
        if standards:
            if self.standards is None:
                raise ValueError("No standards to normalize")
            dataframe = self.standards
        else:
            if self.data is None:
                raise ValueError("No data to normalize")
            dataframe = self.data

        experiments = self.standard_experiments if standards else self.experiments

        dataframe["mu_norm"] = 0
        dataframe["pre_edge"] = 0
        dataframe["post_edge"] = 0

        for experiment in tqdm(experiments, desc="Normalizing data", leave=False):
            experiment_filter = dataframe["Experiment"] == experiment

            for measurement in dataframe["Measurement"][experiment_filter].unique():
                measurement_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"] == measurement
                )
                dummy_group = Group(name="dummy")

                dataframe["mu_norm"][measurement_filter] = dataframe["mu"][
                    measurement_filter
                ] - np.amin(dataframe["mu"][measurement_filter])

                try:
                    pre_edge(
                        dataframe["Energy"][measurement_filter],
                        dataframe["mu_norm"][measurement_filter],
                        group=dummy_group,
                        make_flat=False,
                    )
                    dataframe["pre_edge"][measurement_filter] = dummy_group.pre_edge
                    dataframe["post_edge"][measurement_filter] = dummy_group.post_edge
                    dataframe["mu_norm"][measurement_filter] -= dummy_group.pre_edge
                    pre_edge(
                        dataframe["Energy"][measurement_filter],
                        dataframe["mu_norm"][measurement_filter],
                        group=dummy_group,
                        make_flat=False,
                    )
                    dataframe["mu_norm"][measurement_filter] /= dummy_group.post_edge
                except:
                    dataframe.drop(dataframe[measurement_filter].index, inplace=True)
                    print(
                        f"Error normalizing {experiment} measurement {measurement}. Measurement removed."
                    )

        if standards:
            self.standards = dataframe
        else:
            self.data = dataframe
        return None

    def load_data(
        self,
        average: Union[bool, str] = False,
        measurements_to_average: Union[str, list[int], np.ndarray, range] = "all",
        n_periods: Union[None, int] = None,
        period: Union[None, int] = None,
    ) -> None:
        """
        Load data, apply corrections, and normalize data.

        Args:
            average (Union[bool, str], optional): Whether to average data in a standard or periodic manner. Defaults to False.
            measurements_to_average (Union[str, list[int], np.ndarray, range], optional): Measurements to average. Defaults to "all".
            n_periods (Union[None, int], optional): Number of periods to group measurements in for averaging. Will determine the number of measurements per period automatically. Defaults to None.
            period (Union[None, int], optional): Number of measurements to group for averaging. Will determine the number of periods automatically. Defaults to None.

        Raises:
            ValueError: Invalid average. Must be "standard" or "periodic".

        Returns:
            None: Function does not return anything.
        """
        self._read_data()
        self.experiments = list(self.data["Experiment"].unique())
        self.metals = list(self.data["Metal"].unique())
        self._energy_correction()
        if average:
            if average.lower() == "standard":
                self._average_data(measurements_to_average=measurements_to_average)
            elif average.lower() == "periodic":
                self._average_data_periodic(period=period, n_periods=n_periods)
            else:
                raise ValueError('Invalid average. Must be "standard" or "periodic".')
        self._normalize_data()
        return None

    def load_standards(
        self,
        average: Union[bool, str] = False,
        measurements_to_average: Union[str, list[int], np.ndarray, range] = "all",
        n_periods: Union[None, int] = None,
        period: Union[None, int] = None,
    ) -> None:
        """
        Load experimental standards and store in DataFrame.

        Raises:
            NotImplementedError: Standard loading not implemented yet.

        Returns:
            None: Function does not return anything.
        """
        # raise NotImplementedError("Standard loading not implemented yet")

        self._read_data(standards=True)
        self.standard_experiments = list(self.standards["Experiment"].unique())
        self._energy_correction(standards=True)
        if average:
            if average.lower() == "standard":
                self._average_data(
                    measurements_to_average=measurements_to_average, standards=True
                )
            elif average.lower() == "periodic":
                self._average_data_periodic(
                    period=period, n_periods=n_periods, standards=True
                )
            else:
                raise ValueError('Invalid average. Must be "standard" or "periodic".')
        self._normalize_data(standards=True)
        return None

    def _linear_combination(
        self, weights: Parameters, components: list[np.array]
    ) -> np.array:
        """
        Calculate the linear combination of components using the given weights.

        Args:
            weights (Parameters): Weights for each component.
            components (list[np.array]): List of components to combine.

        Returns:
            np.array: Linear combination of components.
        """
        weights = np.array(list(weights.valuesdict().values()))
        components = np.array(components)
        return np.dot(weights, components)

    def _residual(self, target: np.array, combination: np.array) -> np.array:
        """
        Calculate the residual between the target and the linear combination.

        Args:
            target (np.array): Target data.
            combination (np.array): Linear combination of components.

        Returns:
            np.array: Residual between target and linear combination.
        """
        return target - combination

    def _fit_function(
        self, weights: Parameters, components: list[np.array], target: np.array
    ) -> np.array:
        """
        Fit function for the linear combination of components.

        Args:
            weights (Parameters): Weights for each component.
            components (list[np.array]): List of components to combine.
            target (np.array): Target data.

        Returns:
            np.array: Residual between target and linear combination.
        """
        combination = self._linear_combination(weights, components)
        return self._residual(target, combination)

    # Analysis functions

    def LCA(
        self,
        use_standards: bool = False,
        components: Union[list[int], list[str], None] = [0, -1],
        fit_range: Union[None, tuple[float, float], list[tuple[float, float]]] = None,
    ) -> None:
        """
        Perform Linear Combination Analysis (LCA) on the data.

        Args:
            use_standards (bool, optional): Whether to use standards for LCA. Defaults to False.
            components (Union[list[int], list[str], None], optional): Components to use for LCA. Defaults to [0, -1].
            fit_range (Union[None, tuple[float, float], list[tuple[float, float]]], optional): Energy range to use for fitting. Defaults to None.

        Raises:
            ValueError: No standards loaded.
            ValueError: Number of fit ranges must match number of metals.
            NotImplementedError: LCA with standards not implemented yet.
            ValueError: At least 2 components are required to perform LCA.

        Returns:
            None: Function does not return anything.
        """
        if use_standards:
            if self.standards is None:
                raise ValueError("No standards loaded")

        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError("Number of fit ranges must match number of metals")
        elif isinstance(fit_range, tuple):
            fit_range = [fit_range] * len(self.metals)
        else:
            fit_range = [(0, np.inf)] * len(self.metals)

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

        for i, metal in enumerate(
            tqdm(self.metals, desc="Performing LCA", leave=False)
        ):
            fit_range_filter = (self.data["Energy"] >= fit_range[i][0]) & (
                self.data["Energy"] <= fit_range[i][1]
            )

            if use_standards:
                component_measurements = []
                component_names = []
                for standard_experiment in self.standards["Experiment"][
                    self.standards["Metal"] == metal
                ].unique():
                    measurements = self.standards["Measurement"][
                        self.standards["Experiment"] == standard_experiment
                    ].unique()
                    for measurement in measurements:
                        component_measurements.append(
                            (standard_experiment, measurement)
                        )
                        if len(measurements) == 1:
                            component_names.append(f"{standard_experiment}")
                        else:
                            component_names.append(
                                f"{standard_experiment} ({measurement})"
                            )
                n_components = len(component_measurements)
                if n_components < 2:
                    raise ValueError(
                        "At least 2 components are required to perform LCA."
                    )
                components_mu = []
                for standard_name, measurement in component_measurements:
                    measurement_filter = (
                        (self.standards["Experiment"] == standard_name)
                        & (self.standards["Measurement"] == measurement)
                        & fit_range_filter
                    )
                    components_mu.append(
                        self.standards["mu_norm"][measurement_filter].to_numpy()
                    )
            else:
                component_measurements = self.data["Measurement"][
                    self.data["Metal"] == metal
                ].unique()[components]
                component_names = [
                    f"Measurement {meas_num}" for meas_num in component_measurements
                ]
                n_components = len(component_measurements)
                if n_components < 2:
                    raise ValueError(
                        "At least 2 components are required to perform LCA."
                    )
                components_mu = []
                for measurement in component_measurements:
                    measurement_filter = (
                        (self.data["Metal"] == metal)
                        & (self.data["Measurement"] == measurement)
                        & fit_range_filter
                    )
                    components_mu.append(
                        self.data["mu_norm"][measurement_filter].to_numpy()
                    )

            for measurement in self.data["Measurement"][
                self.data["Metal"] == metal
            ].unique():
                measurement_filter = (
                    (self.data["Metal"] == metal)
                    & (self.data["Measurement"] == measurement)
                    & fit_range_filter
                )
                target = self.data["mu_norm"][measurement_filter].to_numpy()

                weights = Parameters()
                for i in range(1, n_components):
                    weights.add(f"w{i}", value=1 / n_components, min=0, max=1)
                weights.add(
                    f"w{n_components}",
                    min=0,
                    max=1,
                    expr="1 - " + " - ".join([f"w{i}" for i in range(1, n_components)]),
                )

                fit_output = minimize(
                    self._fit_function, weights, args=(components_mu, target)
                )

                # Store data used for LCA
                list_experiment.append(
                    self.data["Experiment"][measurement_filter].values[0]
                )
                list_metal.append(metal)
                list_measurement.append(measurement)
                list_temperature.append(
                    self.data["Temperature"][measurement_filter].values[0]
                )
                list_fit_range.append(fit_range[i])
                list_energy.append(self.data["Energy"][measurement_filter].to_numpy())
                list_component_name.append(f"(Ref) Measurement {measurement}")
                list_component.append(target)
                list_parameter_name.append(None)
                list_parameter_value.append(None)
                list_parameter_error.append(None)
                # Store LCA results
                for j, (name, param) in enumerate(fit_output.params.items()):
                    list_experiment.append(
                        self.data["Experiment"][measurement_filter].values[0]
                    )
                    list_metal.append(metal)
                    list_measurement.append(measurement)
                    list_temperature.append(
                        self.data["Temperature"][measurement_filter].values[0]
                    )
                    list_fit_range.append(fit_range[i])
                    list_energy.append(
                        self.data["Energy"][measurement_filter].to_numpy()
                    )
                    list_component_name.append(component_names[j])
                    list_component.append(components_mu[j])
                    list_parameter_name.append(name)
                    list_parameter_value.append(param.value)
                    if param.stderr is not None:
                        list_parameter_error.append(param.stderr)
                    else:
                        list_parameter_error.append(1e-6)

        self.LCA_result = pd.DataFrame(
            {
                "Experiment": list_experiment,
                "Metal": list_metal,
                "Measurement": list_measurement,
                "Temperature": list_temperature,
                "Fit Range": list_fit_range,
                "Energy": list_energy,
                "Component Name": list_component_name,
                "Component": list_component,
                "Parameter Name": list_parameter_name,
                "Parameter Value": list_parameter_value,
                "Parameter Error": list_parameter_error,
            }
        )
        return None

    def PCA(
        self,
        n_components: Union[None, str, float, int, list] = None,
        fit_range: Union[None, tuple[float, float], list[tuple[float, float]]] = None,
        seed: Union[None, int] = None,
    ) -> None:
        """
        Perform Principal Component Analysis (PCA) on the data.

        Args:
            n_components (Union[None, str, float, int, list], optional): Number of components to keep or fraction of variance to be explained by components. Defaults to None.
            fit_range (Union[None, tuple[float, float], list[tuple[float, float]]], optional): Energy range to use for fitting. Defaults to None.
            seed (Union[None, int], optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: Length of list must match number of experiments.

        Returns:
            None: Function does not return anything.
        """
        if isinstance(n_components, list):
            if len(n_components) != len(self.experiments):
                raise ValueError("Length of list must match number of experiments")
        else:
            n_components = [n_components] * len(self.experiments)

        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError("Number of fit ranges must match number of metals")
        elif isinstance(fit_range, tuple):
            fit_range = [fit_range] * len(self.metals)
        else:
            fit_range = [(0, np.inf)] * len(self.metals)

        experiment_list = []
        metal_list = []
        measurement_list = []
        fit_range_list = []
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

        for i, (experiment, n_components) in enumerate(
            zip(self.experiments, n_components)
        ):

            fit_range_filter = (self.data["Energy"] >= fit_range[i][0]) & (
                self.data["Energy"] <= fit_range[i][1]
            )

            measurements = self.data["Measurement"][
                self.data["Experiment"] == experiment
            ].unique()
            data = (
                self.data["mu_norm"][
                    (self.data["Experiment"] == experiment) & fit_range_filter
                ]
                .to_numpy()
                .reshape(len(measurements), -1)
            )

            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(data)
            pca_weights = pca.transform(data)

            # Store PCA results
            for j, component in enumerate(pca.components_):
                for measurement in measurements:
                    experiment_list.append(experiment)
                    metal_list.append(
                        self.data["Metal"][
                            self.data["Experiment"] == experiment
                        ].values[0]
                    )
                    measurement_list.append(measurement)
                    fit_range_list.append(fit_range[i])
                    n_components_list.append(pca.n_components_)
                    pca_mean_list.append(pca.mean_)
                    explained_variance_list.append(pca.explained_variance_[j])
                    explained_variance_ratio_list.append(
                        pca.explained_variance_ratio_[j]
                    )
                    cumulative_explained_variance_list.append(
                        pca.explained_variance_ratio_[: j + 1].sum()
                    )
                    energy_list.append(
                        self.data["Energy"][
                            (self.data["Experiment"] == experiment) & fit_range_filter
                        ].to_numpy()
                    )
                    component_list.append(component)
                    component_names_list.append(f"PC {j+1}")
                    component_number_list.append(j + 1)
                    weights_list.append(pca_weights[measurement - 1, j])

        self.PCA_result = pd.DataFrame(
            {
                "Experiment": experiment_list,
                "Metal": metal_list,
                "Measurement": measurement_list,
                "Fit Range": fit_range_list,
                "n_components": n_components_list,
                "PCA Mean": pca_mean_list,
                "Explained Variance": explained_variance_list,
                "Explained Variance Ratio": explained_variance_ratio_list,
                "Cumulative Explained Variance": cumulative_explained_variance_list,
                "Energy": energy_list,
                "Component": component_list,
                "Component Name": component_names_list,
                "Component Number": component_number_list,
                "Weight": weights_list,
            }
        )
        return None

    def NMF(
        self,
        n_components: Union[None, str, float, int, list] = None,
        change_cutoff: float = 0.25,
        fit_range: Union[None, tuple[float, float], list[tuple[float, float]]] = None,
        seed: Union[None, int] = None,
    ) -> None:
        """
        Perform Non-negative Matrix Factorization (NMF) on the data.

        Args:
            n_components (Union[None, str, float, int, list], optional): Number of components to keep. Defaults to None.
            change_cutoff (float, optional): Minimum change in reconstruction error to determine number of components. Defaults to 0.25.
            fit_range (Union[None, tuple[float, float], list[tuple[float, float]]], optional): Energy range to use for fitting. Defaults to None.
            seed (Union[None, int], optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: Length of list must match number of experiments.

        Returns:
            None: Function does not return anything.
        """
        if isinstance(n_components, list):
            if len(n_components) != len(self.experiments):
                raise ValueError("Length of list must match number of experiments")
        elif n_components is None:
            n_components = self._determine_NMF_components(
                change_cutoff=change_cutoff, fit_range=fit_range
            )
        else:
            n_components = [n_components] * len(self.experiments)

        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError("Number of fit ranges must match number of metals")
        elif isinstance(fit_range, tuple):
            fit_range = [fit_range] * len(self.metals)
        else:
            fit_range = [(0, np.inf)] * len(self.metals)

        experiment_list = []
        metal_list = []
        measurement_list = []
        fit_range_list = []
        n_components_list = []
        energy_list = []
        component_list = []
        component_names_list = []
        component_number_list = []
        weights_list = []

        for i, (experiment, n_components) in enumerate(
            zip(self.experiments, n_components)
        ):

            fit_range_filter = (self.data["Energy"] >= fit_range[i][0]) & (
                self.data["Energy"] <= fit_range[i][1]
            )

            measurements = self.data["Measurement"][
                self.data["Experiment"] == experiment
            ].unique()
            data = (
                self.data["mu_norm"][
                    (self.data["Experiment"] == experiment) & fit_range_filter
                ]
                .to_numpy()
                .reshape(len(measurements), -1)
            )
            # Remove negative values
            data -= data.min()

            nmf = NMF(n_components=n_components, random_state=seed, init="nndsvda")
            nmf.fit(data)
            nmf_weights = nmf.transform(data)

            # Store NMF results
            for j, component in enumerate(nmf.components_):
                for measurement in measurements:
                    experiment_list.append(experiment)
                    metal_list.append(
                        self.data["Metal"][
                            self.data["Experiment"] == experiment
                        ].values[0]
                    )
                    measurement_list.append(measurement)
                    fit_range_list.append(fit_range[i])
                    n_components_list.append(nmf.n_components_)
                    energy_list.append(
                        self.data["Energy"][
                            (self.data["Experiment"] == experiment) & fit_range_filter
                        ].to_numpy()
                    )
                    component_list.append(component)
                    component_names_list.append(f"Component {j+1}")
                    component_number_list.append(j + 1)
                    weights_list.append(nmf_weights[measurement - 1, j])

        self.NMF_result = pd.DataFrame(
            {
                "Experiment": experiment_list,
                "Metal": metal_list,
                "Measurement": measurement_list,
                "Fit Range": fit_range_list,
                "n_components": n_components_list,
                "Energy": energy_list,
                "Component": component_list,
                "Component Name": component_names_list,
                "Component Number": component_number_list,
                "Weight": weights_list,
            }
        )

        return None

    def _determine_NMF_components(
        self,
        change_cutoff: int,
        fit_range: Union[None, tuple[float, float], list[tuple[float, float]]] = None,
    ) -> list[int]:
        """
        Determine the number of components to use for Non-negative Matrix Factorization (NMF).

        Args:
            change_cutoff (int): Minimum change in reconstruction error to determine number of components.
            fit_range (Union[None, tuple[float, float], list[tuple[float, float]]], optional): Energy range to use for fitting. Defaults to None.

        Returns:
            list[int]: Number of components to use for NMF.
        """

        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError("Number of fit ranges must match number of metals")
        elif isinstance(fit_range, tuple):
            fit_range = [fit_range] * len(self.metals)
        else:
            fit_range = [(0, np.inf)] * len(self.metals)

        experiment_list = []
        metal_list = []
        n_components_list = []
        reconstruction_error_list = []

        for i, experiment in enumerate(self.experiments):

            fit_range_filter = (self.data["Energy"] >= fit_range[i][0]) & (
                self.data["Energy"] <= fit_range[i][1]
            )

            measurements = self.data["Measurement"][
                self.data["Experiment"] == experiment
            ].unique()
            data = np.clip(
                self.data["mu_norm"][
                    (self.data["Experiment"] == experiment) & fit_range_filter
                ]
                .to_numpy()
                .reshape(len(measurements), -1),
                a_min=0,
                a_max=None,
            )

            for n_components in range(len(measurements)):
                nmf = NMF(n_components=n_components + 1)
                nmf.fit(data)

                # Log reconstruction error
                experiment_list.append(experiment)
                metal_list.append(
                    self.data["Metal"][self.data["Experiment"] == experiment].values[0]
                )
                n_components_list.append(n_components + 1)
                reconstruction_error_list.append(nmf.reconstruction_err_)

        self.NMF_component_results = pd.DataFrame(
            {
                "Experiment": experiment_list,
                "Metal": metal_list,
                "n_components": n_components_list,
                "Reconstruction Error": reconstruction_error_list,
            }
        )

        nmf_k = []
        error_change_list = []
        for experiment in self.experiments:
            _nmf = self.NMF_component_results[
                self.NMF_component_results["Experiment"] == experiment
            ]
            error_change = np.absolute(
                np.gradient(_nmf["Reconstruction Error"], _nmf["n_components"])
            )
            error_change_list.extend(error_change)
            k = _nmf["n_components"][error_change > np.abs(change_cutoff)].max()
            nmf_k.append(k)

        self.NMF_component_results["Error Change"] = error_change_list

        return nmf_k

    # Data export functions

    def to_csv(
        self,
        data: str = "data",
        filename: Union[None, str] = None,
        directory: Union[None, str] = None,
        columns: Union[str, list[str]] = None,
    ):
        """
        Export data to a CSV file.

        Args:
            filename (str): Name of the file to save.
            directory (Union[None, str], optional): Directory to save the file. Defaults to None.
            columns (Union[None, list[str]], optional): Columns to export. Defaults to None.

        Returns:
            None: Function does not return anything.
        """

        if data not in ["data", "standards", "LCA", "PCA", "NMF"]:
            raise ValueError("Invalid data type")

        if directory is None:
            directory = self.save_directory

        if data == "data":
            dataframe = self.data
        elif data == "standards":
            dataframe = self.standards
        elif data == "LCA":
            dataframe = self.LCA_result
        elif data == "PCA":
            dataframe = self.PCA_result
        elif data == "NMF":
            dataframe = self.NMF_result

        # TODO: Unravel nested columns before saving

        dataframe.to_csv(directory + filename, index=False, columns=columns)
        return None

    def to_athena(
        self,
        filename: str,
        directory: str = "./",
        columns: Union[None, list[str]] = None,
    ):
        # with open(directory + filename + '.nor', 'w') as file:
        #     file.write('# Exported from autoXAS')
        raise NotImplementedError("Athena export not implemented yet")
        return None

    # Plotting functions

    def plot_data(
        self,
        experiment: Union[str, int] = 0,
        standards: Union[None, str, int] = None,
        save: bool = False,
        filename: str = "data",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot the normalized data for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            standards (Union[None, list[str]], optional): Standards to plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "data".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Standards not implemented yet.

        Returns:
            None: Function does not return anything.
        """
        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        if isinstance(standards, int):
            standards = [self.standard_experiments[standards]]
        elif isinstance(standards, str):
            if len(standards) == 1 or len(standards) == 2:
                standards = self.standards["Experiment"][
                    self.standards["Metal"] == standards
                ].unique()
            elif len(standards) > 2:
                if standards not in self.standard_experiments:
                    raise ValueError("Invalid standard name")
                else:
                    standards = [standards]

        n_measurements = int(
            self.data["Measurement"][self.data["Experiment"] == experiment].max()
        )
        experiment_filter = self.data["Experiment"] == experiment

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment
            fig = px.line(
                data_frame=self.data[experiment_filter],
                x="Energy",
                y="mu_norm",
                color="Measurement",
                color_discrete_sequence=px.colors.sample_colorscale(
                    "viridis", samplepoints=n_measurements
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

            if standards:
                i_standard = 0
                for standard in standards:
                    experiment_filter = self.standards["Experiment"] == standard
                    measurements = self.standards["Measurement"][
                        experiment_filter
                    ].unique()
                    for measurement in measurements:
                        standard_filter = (self.standards["Experiment"] == standard) & (
                            self.standards["Measurement"] == measurement
                        )
                        if len(measurements) == 1:
                            line_name = f"{standard}"
                        else:
                            line_name = f"{standard} ({measurement})"
                        fig.add_scatter(
                            x=self.standards["Energy"][standard_filter],
                            y=self.standards["mu_norm"][standard_filter],
                            mode="lines",
                            name=line_name,
                            line=dict(
                                width=2,
                                dash="dash",
                                color=px.colors.qualitative.Safe[i_standard],
                            ),
                        )
                        i_standard += 1

            # Specify title text
            if show_title:
                title_text = f"<b>Normalized data<br><sup><i>{experiment}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title="<b>Normalized [a.u.]</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            # Create figure object and set the figure size
            plt.figure(figsize=(10, 8))

            # Plot all measurements of specified experiment
            sns.lineplot(
                data=self.data[experiment_filter],
                x="Energy",
                y="mu_norm",
                hue="Measurement",
                palette="viridis",
            )
            # Set limits of x-axis to match the edge measurements
            plt.xlim(
                (
                    np.amin(self.data["Energy"][experiment_filter]),
                    np.amax(self.data["Energy"][experiment_filter]),
                )
            )
            # Specify text and formatting of axis labels
            plt.xlabel(f"Energy [{self.energy_unit}]", fontsize=14, fontweight="bold")
            plt.ylabel("Normalized", fontsize=14, fontweight="bold")
            # Specify placement, formatting and title of the legend
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="Measurement",
                fontsize=12,
                title_fontsize=13,
                ncol=1,
            )
            # Enforce matplotlibs tight layout
            plt.tight_layout()
            if save:
                plt.savefig(self.save_directory + "figures/" + filename + format)

            if show:
                plt.show()
        return None

    def plot_temperature_curves(
        self,
        save: bool = False,
        filename: str = "temperature_curve",
        format: str = ".png",
        show: bool = True,
        show_title: bool = True,
    ):
        """
        Plot the temperature curves for all experiments.

        Args:
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "temperature_curve".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = "%{y:.1f} +/- %{customdata[0]:.1f} " + self.temperature_unit
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = line(
                data_frame=self.data,
                x="Measurement",
                y="Temperature",
                error_y="Temperature (std)",
                error_y_mode="band",
                custom_data=["Temperature (std)"],
                color="Experiment",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
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
                title_text = f"<b>Temperature curves</b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Temperature [{self.temperature_unit}]</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()
        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_waterfall(
        self,
        experiment: Union[str, int] = 0,
        y_axis: str = "Measurement",
        vmin: Union[None, float] = None,
        vmax: Union[None, float] = None,
        save: bool = False,
        filename: str = "waterfall",
        format: str = ".png",
        show: bool = True,
        show_title: bool = True,
    ):
        """
        Plot a waterfall plot for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            y_axis (str, optional): Column to use as Y-axis in the plot. Defaults to "Measurement".
            vmin (Union[None, float], optional): Minimum value for the color scale. Defaults to None.
            vmax (Union[None, float], optional): Maximum value for the color scale. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "waterfall".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        experiment_filter = self.data["Experiment"] == experiment

        heatmap_data = self.data[experiment_filter].pivot(
            index=y_axis, columns="Energy", values="mu_norm"
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".2f"

            # Plot the measurements of the selected experiment
            fig = px.imshow(
                img=heatmap_data,
                zmin=vmin,
                zmax=vmax,
                origin="lower",
                color_continuous_scale="viridis",
                aspect="auto",
                labels=dict(color="<b>Normalized [a.u.]</b>"),
            )

            # Specify title text
            if show_title:
                title_text = (
                    f"<b><i>In situ</i> overview<br><sup><i>{experiment}</i></sup></b>"
                )
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title=f"<b>{y_axis}</b>",
                font=dict(
                    size=14,
                ),
                hovermode="closest",
                coloraxis=dict(
                    colorbar=dict(
                        titleside="right",
                    ),
                ),
            )

            hovertemplate = f"Measurement: %{{y}}<br>Energy: %{{x:{x_formatting}}} {self.energy_unit}<br>Normalized: %{{z:.2f}}<extra></extra>"
            fig.update_traces(hovertemplate=hovertemplate)

            # Add and format spikes
            fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()
        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_change(
        self,
        experiment: Union[str, int] = 0,
        reference_measurement: int = 1,
        y_axis: str = "Measurement",
        vmin: Union[None, float] = None,
        vmax: Union[None, float] = None,
        save: bool = False,
        filename: str = "change",
        format: str = ".png",
        show: bool = True,
        show_title: bool = True,
    ):
        """
        Plot the change in normalized data for a given experiment

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            reference_measurement (int, optional): Measurement to use as reference. Defaults to 1.
            y_axis (str, optional): Column to use as Y-axis in the plot. Defaults to "Measurement".
            vmin (Union[None, float], optional): Minimum value for the color scale. Defaults to None.
            vmax (Union[None, float], optional): Maximum value for the color scale. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "change".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        experiment_filter = self.data["Experiment"] == experiment
        experiment_data = self.data[experiment_filter]
        reference = experiment_data["mu_norm"][
            (self.data["Measurement"] == reference_measurement)
        ].to_numpy()
        difference_from_reference = (
            experiment_data["mu_norm"].to_numpy().reshape((-1, reference.shape[0]))
            - reference
        )
        experiment_data["Difference from reference"] = (
            difference_from_reference.reshape((-1))
        )

        heatmap_data = experiment_data.pivot(
            index=y_axis, columns="Energy", values="Difference from reference"
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".2f"

            # Plot the measurements of the selected experiment
            fig = px.imshow(
                img=heatmap_data,
                zmin=vmin,
                zmax=vmax,
                origin="lower",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0.0,
                aspect="auto",
                labels=dict(color="<b>\u0394 Normalized [a.u.]</b>"),
            )

            fig.add_hline(
                y=experiment_data[y_axis].unique()[reference_measurement - 1],
                line_color="black",
                line_width=2,
                annotation_text=f"<b>Reference (Measurement {reference_measurement})</b>",
                annotation_position="top left",
                annotation_font=dict(
                    color="black",
                    size=11,
                ),
            )

            # Specify title text
            if show_title:
                title_text = (
                    f"<b><i>In situ</i> changes<br><sup><i>{experiment}</i></sup></b>"
                )
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title=f"<b>{y_axis}</b>",
                font=dict(
                    size=14,
                ),
                hovermode="closest",
                coloraxis=dict(
                    colorbar=dict(
                        titleside="right",
                    ),
                ),
            )

            hovertemplate = f"Measurement: %{{y}}<br>Energy: %{{x:{x_formatting}}} {self.energy_unit}<br>\u0394 Normalized: %{{z:.2f}}<extra></extra>"
            fig.update_traces(hovertemplate=hovertemplate)

            # Add and format spikes
            fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
            fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()
        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_LCA(
        self,
        experiment: Union[str, int] = 0,
        save: bool = False,
        filename: str = "LCA",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot the results of Linear Combination Analysis (LCA) for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "LCA".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        experiment_filter = self.LCA_result["Experiment"] == experiment

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = (
                f"%{{y:{hover_format}}} +/- %{{customdata[0]:{hover_format}}}"
            )
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            df_data = self.LCA_result[experiment_filter]
            # df_data["Clipped Parameter Error"] = df_data["Parameter Error"].clip(
            #     upper=1
            # )
            fig = line(
                data_frame=df_data,
                x="Measurement",
                y="Parameter Value",
                error_y="Parameter Error",
                error_y_mode="band",
                custom_data=["Parameter Error"],
                color="Parameter Name",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
                labels={"Parameter Name": "Component"},
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
                title_text = f"<b>Linear Combination Analysis<br><sup><i>{experiment}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                yaxis_range=[-0.5, 1.5],
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_LCA_frame(
        self,
        experiment: Union[str, int] = 0,
        measurement: int = 1,
        save: bool = False,
        filename: str = "LCA_frame",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot a single frame of the Linear Combination Analysis (LCA) for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            measurement (int): Measurement to plot. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "LCA_frame".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        data_filter = (self.LCA_result["Experiment"] == experiment) & (
            self.LCA_result["Measurement"] == measurement
        )

        reference_data_filter = (self.data["Experiment"] == experiment) & (
            self.data["Measurement"] == measurement
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot reference measurement
            reference_data = self.data["mu_norm"][reference_data_filter].to_numpy()
            reference = self.LCA_result["Component"][
                (
                    self.LCA_result["Component Name"]
                    == f"(Ref) Measurement {measurement}"
                )
                & data_filter
            ].values[0]

            fig = go.Figure(
                go.Scatter(
                    x=self.data["Energy"][reference_data_filter].to_numpy(),
                    y=reference_data,
                    name="Data",
                    mode="lines",
                    line=dict(
                        color="black",
                    ),
                )
            )

            linear_combination = np.zeros_like(reference)
            # Plot components
            for i, component in enumerate(
                self.LCA_result["Component Name"][data_filter].unique()
            ):
                if "(Ref)" in component:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=self.LCA_result["Energy"][
                            (self.LCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        y=self.LCA_result["Component"][
                            (self.LCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0]
                        * self.LCA_result["Parameter Value"][
                            (self.LCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        name=f"Component {self.LCA_result['Parameter Name'][(self.LCA_result['Component Name'] == component) & data_filter].values[0][1:]}",
                        mode="lines",
                        line=dict(
                            color=sns.color_palette("colorblind").as_hex()[i],
                        ),
                    )
                )

                linear_combination += (
                    self.LCA_result["Component"][
                        (self.LCA_result["Component Name"] == component) & data_filter
                    ].values[0]
                    * self.LCA_result["Parameter Value"][
                        (self.LCA_result["Component Name"] == component) & data_filter
                    ].values[0]
                )

            # Plot linear combination
            fig.add_trace(
                go.Scatter(
                    x=self.LCA_result["Energy"][
                        (
                            self.LCA_result["Component Name"]
                            == f"(Ref) Measurement {measurement}"
                        )
                        & data_filter
                    ].values[0],
                    y=linear_combination,
                    name="Linear combination",
                    mode="lines",
                    line=dict(
                        color="magenta",
                    ),
                )
            )

            # Plot residuals
            fig.add_trace(
                go.Scatter(
                    x=self.LCA_result["Energy"][
                        (
                            self.LCA_result["Component Name"]
                            == f"(Ref) Measurement {measurement}"
                        )
                        & data_filter
                    ].values[0],
                    y=reference - linear_combination,
                    name="Residual",
                    mode="lines",
                    line=dict(
                        color="red",
                    ),
                )
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
                title_text = f"<b>Linear Combination Analysis<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title="<b>Normalized [a.u.]</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_LCA_comparison(
        self,
        component: int = 1,
        save: bool = False,
        filename: str = "LCA_comparison",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot comparison of Linear Combination Analysis (LCA) components across experiments.

        Args:
            component (int): Components to compare. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "LCA_comparison".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = (
                f"%{{y:{hover_format}}} +/- %{{customdata[0]:{hover_format}}}"
            )
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = line(
                data_frame=self.LCA_result[
                    self.LCA_result["Parameter Name"] == f"w{component}"
                ],
                x="Measurement",
                y="Parameter Value",
                error_y="Parameter Error",
                error_y_mode="band",
                custom_data=["Parameter Error"],
                color="Metal",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
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
                title_text = f"<b>LCA Transition Comparison<br><sup><i>Component {component}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                yaxis_range=[-0.5, 1.5],
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_PCA(
        self,
        experiment: Union[str, int] = 0,
        save: bool = False,
        filename: str = "PCA",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot the results of Principal Component Analysis (PCA) for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "PCA".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        experiment_filter = self.PCA_result["Experiment"] == experiment

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = px.line(
                data_frame=self.PCA_result[experiment_filter],
                x="Measurement",
                y="Weight",
                color="Component Name",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
                labels={"Parameter Name": "Component"},
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
                title_text = f"<b>Principal Component Analysis<br><sup><i>{experiment}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")
        return None

    def plot_PCA_frame(
        self,
        experiment: Union[str, int] = 0,
        measurement: int = 1,
        save: bool = False,
        filename: str = "PCA_frame",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot a single frame of the Principal Component Analysis (PCA) for a given experiment

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            measurement (int): Measurement to plot. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "PCA_frame".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        data_filter = (self.PCA_result["Experiment"] == experiment) & (
            self.PCA_result["Measurement"] == measurement
        )

        reference_data_filter = (self.data["Experiment"] == experiment) & (
            self.data["Measurement"] == measurement
        )

        fit_range = self.PCA_result["Fit Range"][data_filter].values[0]
        fit_range_filter = (self.data["Energy"] >= fit_range[0]) & (
            self.data["Energy"] <= fit_range[1]
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot reference measurement
            reference_data = self.data["mu_norm"][reference_data_filter].to_numpy()
            reference = self.data["mu_norm"][
                reference_data_filter & fit_range_filter
            ].to_numpy()

            fig = go.Figure(
                go.Scatter(
                    x=self.data["Energy"][reference_data_filter].to_numpy(),
                    y=reference_data,
                    name="Data",
                    mode="lines",
                    line=dict(
                        color="black",
                    ),
                )
            )

            # Plot PCA mean
            pca_mean = self.PCA_result["PCA Mean"][data_filter].values[0]

            fig.add_trace(
                go.Scatter(
                    x=self.PCA_result["Energy"][
                        (self.PCA_result["Component Name"] == f"PC 1") & data_filter
                    ].values[0],
                    y=pca_mean,
                    name="PCA mean",
                    mode="lines",
                    line=dict(
                        color="grey",
                    ),
                )
            )

            pca_reconstruction = np.zeros_like(reference)

            # Plot components
            for i, component in enumerate(
                self.PCA_result["Component Name"][data_filter].unique()
            ):
                fig.add_trace(
                    go.Scatter(
                        x=self.PCA_result["Energy"][
                            (self.PCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        y=self.PCA_result["Component"][
                            (self.PCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0]
                        * self.PCA_result["Weight"][
                            (self.PCA_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        name=component,
                        mode="lines",
                        line=dict(
                            color=sns.color_palette("colorblind").as_hex()[i],
                        ),
                    )
                )

                pca_reconstruction += (
                    self.PCA_result["Component"][
                        (self.PCA_result["Component Name"] == component) & data_filter
                    ].values[0]
                    * self.PCA_result["Weight"][
                        (self.PCA_result["Component Name"] == component) & data_filter
                    ].values[0]
                )

            pca_reconstruction += pca_mean

            # Plot PCA reconstruction
            fig.add_trace(
                go.Scatter(
                    x=self.PCA_result["Energy"][
                        (self.PCA_result["Component Name"] == f"PC 1") & data_filter
                    ].values[0],
                    y=pca_reconstruction,
                    name="PCA reconstruction",
                    mode="lines",
                    line=dict(
                        color="magenta",
                    ),
                )
            )

            # Plot residual
            fig.add_trace(
                go.Scatter(
                    x=self.PCA_result["Energy"][
                        (self.PCA_result["Component Name"] == f"PC 1") & data_filter
                    ].values[0],
                    y=reference - pca_reconstruction,
                    name="Residual",
                    mode="lines",
                    line=dict(
                        color="red",
                    ),
                )
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
                title_text = f"<b>Principal Component Analysis<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title="<b>Normalized [a.u.]</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

    def plot_PCA_comparison(
        self,
        component: Union[int, list[int]] = 1,
        save: bool = False,
        filename: str = "PCA_comparison",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot comparison of Principal Component Analysis (PCA) components across experiments.

        Args:
            component (Union[int, list[int]]): Components to compare. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "PCA_comparison".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Length of list must match number of experiments.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(component, list):
            if len(component) != len(self.experiments):
                raise ValueError("Length of list must match number of experiments")
        else:
            component = [component] * len(self.experiments)

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = go.Figure()

            for i, experiment in enumerate(self.experiments):
                data_filter = (self.PCA_result["Experiment"] == experiment) & (
                    self.PCA_result["Component Number"] == component[i]
                )

                fig.add_trace(
                    go.Scatter(
                        x=self.PCA_result["Measurement"][data_filter],
                        y=self.PCA_result["Weight"][data_filter],
                        name=self.PCA_result["Metal"][data_filter].values[0]
                        + f" (PC {component[i]})",
                        mode="lines",
                        line=dict(
                            color=sns.color_palette("colorblind").as_hex()[i],
                        ),
                    )
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
                title_text = f"<b>PCA Transition Comparison</b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_PCA_explained_variance(
        self,
        plot_type: str = "cumulative",
        variance_threshold: Union[None, float] = None,
        fit_range: Union[None, tuple[float, float], list[tuple[float, float]]] = None,
        seed: Union[None, int] = None,
        save: bool = False,
        filename: str = "PCA_explained_variance",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2%",
        show_title: bool = True,
    ):
        """
        Plot the explained variance of Principal Component Analysis (PCA) for all experiments.

        Args:
            plot_type (str, optional): Type of plot to generate. Defaults to "cumulative".
            variance_threshold (Union[None, float], optional): Threshold for the explained variance to plot. Defaults to None.
            fit_range (Union[None, tuple[float, float], list[tuple[float, float]]], optional): Range of energies to fit the PCA. Defaults to None.
            seed (Union[None, int], optional): Random seed for PCA. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "PCA_explained_variance".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2%".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid plot type.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if plot_type not in ["ratio", "cumulative"]:
            raise ValueError(
                'Invalid plot type. Choose between "ratio" and "cumulative"'
            )

        if isinstance(fit_range, list):
            if len(fit_range) != len(self.metals):
                raise ValueError("Number of fit ranges must match number of metals")
        elif isinstance(fit_range, tuple):
            fit_range = [fit_range] * len(self.metals)
        else:
            fit_range = [(0, np.inf)] * len(self.metals)

        experiment_list = []
        metal_list = []
        component_number_list = []
        component_name_list = []
        explained_variance_list = []
        explained_variance_ratio_list = []
        cumulative_explained_variance_list = []

        for i, experiment in enumerate(self.experiments):

            fit_range_filter = (self.data["Energy"] >= fit_range[i][0]) & (
                self.data["Energy"] <= fit_range[i][1]
            )

            measurements = self.data["Measurement"][
                self.data["Experiment"] == experiment
            ].unique()
            data = (
                self.data["mu_norm"][
                    (self.data["Experiment"] == experiment) & fit_range_filter
                ]
                .to_numpy()
                .reshape(len(measurements), -1)
            )

            pca = PCA(random_state=seed)
            pca.fit(data)

            # Store PCA results
            for i in range(pca.n_components_ + 1):
                experiment_list.append(experiment)
                metal_list.append(
                    self.data["Metal"][self.data["Experiment"] == experiment].values[0]
                )
                component_number_list.append(i)
                component_name_list.append(f"PC {i}")
                if i == 0:
                    explained_variance_list.append(0)
                    explained_variance_ratio_list.append(0)
                    cumulative_explained_variance_list.append(0)
                else:
                    explained_variance_list.append(pca.explained_variance_[i - 1])
                    explained_variance_ratio_list.append(
                        pca.explained_variance_ratio_[i - 1]
                    )
                    cumulative_explained_variance_list.append(
                        pca.explained_variance_ratio_[:i].sum()
                    )

        pca_results = pd.DataFrame(
            {
                "Experiment": experiment_list,
                "Metal": metal_list,
                "Component Number": component_number_list,
                "Component Name": component_name_list,
                "Explained Variance": explained_variance_list,
                "Explained Variance Ratio": explained_variance_ratio_list,
                "Cumulative Explained Variance": cumulative_explained_variance_list,
            }
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            if plot_type == "ratio":
                fig = px.bar(
                    data_frame=pca_results[pca_results["Component Number"] != 0],
                    x="Component Name",
                    y="Explained Variance Ratio",
                    color="Metal",
                    color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
                    barmode="group",
                )

                # Specify axis titles
                xaxis_title = "<b>Principal Components</b>"
                yaxis_title = "<b>Explained Variance</b>"

                # Specify title text
                if show_title:
                    title_text = f"<b>Explained Variance Ratio</b>"
                else:
                    title_text = ""

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

            elif plot_type == "cumulative":
                fig = px.line(
                    data_frame=pca_results,
                    x="Component Number",
                    y="Cumulative Explained Variance",
                    color="Metal",
                    color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
                )

                # Specify axis titles
                xaxis_title = "<b>Number of Principal Components</b>"
                yaxis_title = "<b>Explained Variance</b>"

                # Specify title text
                if show_title:
                    title_text = f"<b>Cumulative Explained Variance</b>"
                else:
                    title_text = ""

                if variance_threshold:
                    fig.add_hline(
                        y=variance_threshold,
                        line_color="black",
                        line_width=2,
                        line_dash="dash",
                        annotation_text=f"<b>{variance_threshold:.0%}</b>",
                        annotation_position="bottom right",
                        annotation_font=dict(
                            color="black",
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
                title_x=0.5,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
                yaxis=dict(
                    tickformat=".0%",
                ),
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_NMF(
        self,
        experiment: Union[str, int] = 0,
        save: bool = False,
        filename: str = "NMF",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot the results of Non-negative Matrix Factorization (NMF) for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "NMF".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        experiment_filter = self.NMF_result["Experiment"] == experiment

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = px.line(
                data_frame=self.NMF_result[experiment_filter],
                x="Measurement",
                y="Weight",
                color="Component Name",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
                labels={"Parameter Name": "Component"},
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
                title_text = f"<b>Non-negative Matrix Factorization<br><sup><i>{experiment}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_NMF_frame(
        self,
        experiment: Union[str, int] = 0,
        measurement: int = 1,
        save: bool = False,
        filename: str = "NMF_frame",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot a single frame of the Non-negative Matrix Factorization (NMF) for a given experiment.

        Args:
            experiment (Union[str, int]): Experiment to plot. Defaults to 0.
            measurement (int): Measurement to plot. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "NMF_frame".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Invalid experiment name.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(experiment, int):
            experiment = self.experiments[experiment]
        elif isinstance(experiment, str) and experiment not in self.experiments:
            raise ValueError("Invalid experiment name")

        data_filter = (self.NMF_result["Experiment"] == experiment) & (
            self.NMF_result["Measurement"] == measurement
        )

        reference_data_filter = (self.data["Experiment"] == experiment) & (
            self.data["Measurement"] == measurement
        )

        fit_range = self.NMF_result["Fit Range"][data_filter].values[0]
        fit_range_filter = (self.data["Energy"] >= fit_range[0]) & (
            self.data["Energy"] <= fit_range[1]
        )

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot reference measurement
            reference_data = self.data["mu_norm"][reference_data_filter].to_numpy()
            reference = self.data["mu_norm"][
                reference_data_filter & fit_range_filter
            ].to_numpy()

            fig = go.Figure(
                go.Scatter(
                    x=self.data["Energy"][reference_data_filter].to_numpy(),
                    y=reference_data,
                    name="Data",
                    mode="lines",
                    line=dict(
                        color="black",
                    ),
                )
            )

            nmf_reconstruction = np.zeros_like(reference)

            # Plot components
            for i, component in enumerate(
                self.NMF_result["Component Name"][data_filter].unique()
            ):
                fig.add_trace(
                    go.Scatter(
                        x=self.NMF_result["Energy"][
                            (self.NMF_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        y=self.NMF_result["Component"][
                            (self.NMF_result["Component Name"] == component)
                            & data_filter
                        ].values[0]
                        * self.NMF_result["Weight"][
                            (self.NMF_result["Component Name"] == component)
                            & data_filter
                        ].values[0],
                        name=component,
                        mode="lines",
                        line=dict(
                            color=sns.color_palette("colorblind").as_hex()[i],
                        ),
                    )
                )

                nmf_reconstruction += (
                    self.NMF_result["Component"][
                        (self.NMF_result["Component Name"] == component) & data_filter
                    ].values[0]
                    * self.NMF_result["Weight"][
                        (self.NMF_result["Component Name"] == component) & data_filter
                    ].values[0]
                )

            # Plot NMF reconstruction
            fig.add_trace(
                go.Scatter(
                    x=self.NMF_result["Energy"][
                        (self.NMF_result["Component Name"] == f"Component 1")
                        & data_filter
                    ].values[0],
                    y=nmf_reconstruction,
                    name="NMF reconstruction",
                    mode="lines",
                    line=dict(
                        color="magenta",
                    ),
                )
            )

            # Plot residual
            fig.add_trace(
                go.Scatter(
                    x=self.NMF_result["Energy"][
                        (self.NMF_result["Component Name"] == f"Component 1")
                        & data_filter
                    ].values[0],
                    y=reference - nmf_reconstruction,
                    name="Residual",
                    mode="lines",
                    line=dict(
                        color="red",
                    ),
                )
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
                title_text = f"<b>Non-negative Matrix Factorization<br><sup><i>{experiment} - Measurement {measurement}</i></sup></b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title=f"<b>Energy [{self.energy_unit}]</b>",
                yaxis_title="<b>Normalized [a.u.]</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_NMF_comparison(
        self,
        component: Union[int, list[int]] = 1,
        save: bool = False,
        filename: str = "NMF_comparison",
        format: str = ".png",
        show: bool = True,
        hover_format: str = ".2f",
        show_title: bool = True,
    ):
        """
        Plot comparison of Non-negative Matrix Factorization (NMF) components across experiments.

        Args:
            component (Union[int, list[int]]): Components to compare. Defaults to 1.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "NMF_comparison".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            hover_format (str, optional): Format of numbers in the hover text. Defaults to ".2f".
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            ValueError: Length of list must match number of experiments.
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if isinstance(component, list):
            if len(component) != len(self.experiments):
                raise ValueError("Length of list must match number of experiments")
        else:
            component = [component] * len(self.experiments)

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = f"%{{y:{hover_format}}}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = go.Figure()

            for i, experiment in enumerate(self.experiments):
                data_filter = (self.NMF_result["Experiment"] == experiment) & (
                    self.NMF_result["Component Number"] == component[i]
                )

                fig.add_trace(
                    go.Scatter(
                        x=self.NMF_result["Measurement"][data_filter],
                        y=self.NMF_result["Weight"][data_filter],
                        name=self.NMF_result["Metal"][data_filter].values[0]
                        + f" (Component {component[i]})",
                        mode="lines",
                        line=dict(
                            color=sns.color_palette("colorblind").as_hex()[i],
                        ),
                    )
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
                title_text = f"<b>NMF Transition Comparison</b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Measurement</b>",
                yaxis_title=f"<b>Weight</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None

    def plot_NMF_error_change(
        self,
        change_cutoff: float = 0.25,
        save: bool = False,
        filename: str = "NMF_error_change",
        format: str = ".png",
        show: bool = True,
        show_title: bool = True,
    ):
        """
        Plot the change in error of Non-negative Matrix Factorization (NMF) as a function of number of components.

        Args:
            change_cutoff (float, optional): Error change cutoff to plot. Defaults to 0.25.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file to save. Defaults to "NMF_error_change".
            format (str, optional): Format of the file to save. Defaults to ".png".
            show (bool, optional): Whether to show the plot. Defaults to True.
            show_title (bool, optional): Whether to show the title. Defaults to True.

        Raises:
            NotImplementedError: Matplotlib plot not implemented yet.

        Returns:
            None: Function does not return anything.
        """

        if self.interactive:
            # Formatting for hover text
            x_formatting = ".0f"
            hovertemplate = "%{y:.2f}"
            hovermode = "x unified"

            # Plot the measurements of the selected experiment/edge
            fig = px.line(
                data_frame=self.NMF_component_results,
                x="n_components",
                y="Error Change",
                color="Metal",
                color_discrete_sequence=sns.color_palette("colorblind").as_hex(),
            )

            fig.add_hline(
                y=change_cutoff,
                line_color="black",
                line_width=2,
                line_dash="dash",
                annotation_text=f"<b>{change_cutoff:.2f}</b>",
                annotation_position="bottom right",
                annotation_font=dict(
                    color="black",
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

            # Specify title text
            if show_title:
                title_text = f"<b>NMF Error Change</b>"
            else:
                title_text = ""

            # Specify text and formatting of axis labels
            fig.update_layout(
                title=title_text,
                title_x=0.5,
                xaxis_title="<b>Number of NMF components</b>",
                yaxis_title="<b>\u0394 Error</b>",
                font=dict(
                    size=14,
                ),
                hovermode=hovermode,
            )

            if save:
                fig.write_image(self.save_directory + "figures/" + filename + format)

            if show:
                fig.show()

        else:
            raise NotImplementedError("Matplotlib plot not implemented yet")

        return None
