#%% Imports

# Packages for math
import numpy as np
# Packages for typing
from typing import Union
# Packages for handling data
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# Packages for plotting
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
sns.set_theme()
import plotly_express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

#%% Plotting functions

def plot_non_normalized_xas(
    df: pd.DataFrame, 
    experiment: str, 
    transmission: bool=False, 
    pre_edge: bool=False, 
    post_edge: bool=False, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='raw_xas_plot.png'
) -> None:
    """------------------------------------------------------------------
    Plotting of non-normalized XAS data.

    The energy shift from the measurements is corrected in the plotted data.

    Args:
        df (pd.DataFrame): The dataset after normalization.
        experiment (str): The metal, metal foil or precursor that was measured.
        transmission (optional, bool): Boolean flag deciding if absorption (False) or transmission (True) signal is used. Defaults to False.
        pre_edge (optional, bool): Boolean flag controlling if the pre-edge is plotted. Defaults to False.
        post_edge (optional, bool): Boolean flag controlling if the post-edge is plotted. Defaults to False.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'raw_xas_plot.png'.

    Returns:
        None
    """    
    # Check if the experiment exists in the dataset
    assert experiment in df.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {df.Experiment.unique()}'
    # Choose what type of XAS data to work with
    if transmission:
        data_type = 'Transmission'
    else:
        data_type = 'Absorption'
    # Create filter for relevant values
    df_filter = (df['Experiment'] == experiment) & (df['Measurement'] == 1)
    # Extract the minimum value in the data
    min_value = np.amin(df[data_type][df_filter])
    # Extract the number of measurements
    n_measurements = int(np.amax(df['Measurement'][(df['Experiment'] == experiment)]))
    if n_measurements == 1:
        n_measurements += 1
    # Determine what to use as x-axis
    if df['Energy_Corrected'][df_filter].sum() > 0.:
        x_column = 'Energy_Corrected'
    else: 
        x_column = 'Energy'
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(10,8))
        # Plot the measurements of the selected experiment/edge
        sns.lineplot(
            data=df[(df['Experiment'] == experiment)], 
            x=x_column, 
            y=data_type, 
            hue='Measurement', 
            palette='viridis',
            )
        # Plot the pre-edge fit 
        if pre_edge:
            sns.lineplot(
                x=df[x_column][df_filter], 
                y=df['pre_edge'][df_filter] + min_value,  
                color='r',
                linewidth=3,
                label='Pre-edge'
                )
        # Plot the post-edge
        if post_edge:
            sns.lineplot(
                x=df[x_column][df_filter], 
                y=df['post_edge'][df_filter] + min_value,  
                color='k',
                linewidth=3,
                label='Post-edge'
                )
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Energy [eV]', 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            r'X-ray transmission, µ(E)$\cdot$x', 
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
            ncol=1
            )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive figure using plotly
    elif interactive:
        # Formatting of the hover label "title"
        x_formatting = '.0f'
        # Plot the measurements of the selected experiment/edge
        fig = px.line(
            data_frame=df[(df['Experiment'] == experiment)],
            x=x_column,
            y=data_type,
            color='Measurement',
            color_discrete_sequence=px.colors.sample_colorscale('viridis', samplepoints=n_measurements),
        )
        # Change line formatting
        fig.update_traces(
            line=dict(
                width=2,
            ),
            xhoverformat=x_formatting,
        )
        # Plot the pre-edge fit 
        if pre_edge:
            fig.add_trace(
                go.Scatter(
                    x=df[x_column][df_filter], 
                    y=df['pre_edge'][df_filter] + min_value, 
                    mode='lines',
                    name='Pre-edge',
                    line=dict(
                        width=3,
                        color='red',
                    ),
                    xhoverformat=x_formatting,
                ))
        # Plot the post-edge
        if post_edge:
            fig.add_trace(
                go.Scatter(
                    x=df[x_column][df_filter], 
                    y=df['post_edge'][df_filter] + min_value, 
                    mode='lines',
                    name='Post-edge',
                    line=dict(
                        width=3,
                        color='black',
                    ),
                    xhoverformat=x_formatting,
                ))
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Energy [eV]</b>',
            yaxis_title='<b>'+data_type+'</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
        )
        # Customize the hover labels
        hovertemplate = '<br>Absorption = %{y:.2f} <br>Energy = %{x:.0f} eV'
        fig.update_traces(hovertemplate=hovertemplate)
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_data(
    data: pd.DataFrame, 
    metal: str, 
    foils: Union[pd.DataFrame, None]=None, 
    products: Union[pd.DataFrame, None]=None, 
    intermediates: Union[pd.DataFrame, None]=None, 
    precursors: Union[pd.DataFrame, None]=None, 
    precursor_suffix: Union[str, None]=None, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='xas_data.png'
) -> None:
    """------------------------------------------------------------------
    Plotting of normalized XAS data.

    The measured standards can be shown in the same plot if the datasets with metal foils and precursors are provided.

    Args:
        data (pd.DataFrame): The dataset.
        metal (str): The metal that should be plotted.
        foils (optional, Union[pd.DataFrame | None]): Dataset of metal foils. Defaults to None.
        products (optional, Union[pd.DataFrame | None]): Dataset of products. Defaults to None.
        intermediates (optional, Union[pd.DataFrame | None]): Dataset of intermediates. Defaults to None.
        precursors (optional, Union[pd.DataFrame, None]): Dataset of precursors. Defaults to None.
        precursor_suffix (optional, Union[str, None]): The counter-ion used in a specific precursor that should be plotted. Defaults to None.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'xas_data.png'.

    Returns:
        None
    """    
    # Check if metal exists in the dataset
    assert metal in data.Metal.unique(), f'No metal with the name: {metal}\n\nValid values are: {data.Metal.unique()}'
    # Extract the number of measurements
    n_measurements = int(np.amax(data['Measurement']))
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(10,8))
        # Plot metal foil if provided
        if type(foils) != type(None):
            # Check if the metal foil exists in the dataset
            assert metal in foils.Metal.unique(), f'No precursor containing: {metal}\n\nValid values are: {foils.Experiment.unique()}'
            sns.lineplot(
                data=foils[(foils['Metal'] == metal) & (foils['Measurement'] == 1)], 
                x='Energy_Corrected', 
                y='Normalized', 
                color='k', 
                linewidth=3, 
                label=metal+' foil'
                )
        # Plot precursor(s) if provided
        if type(precursors) != type(None):
            # Plot only specific precursor if it is provided
            if type(precursor_suffix) != type(None):
                # Check if the specified precursor exists in the dataset
                assert metal+precursor_suffix in precursors.Experiment.unique(), f'No metal with the name: {metal+precursor_suffix}\n\nValid values are: {precursors.Experiment.unique()}'
                sns.lineplot(
                    data=precursors[(precursors['Experiment'] == metal+precursor_suffix) & (precursors['Measurement'] == 1)], 
                    x='Energy_Corrected', 
                    y='Normalized', 
                    color='r', 
                    linewidth=3, 
                    label=metal+precursor_suffix
                    )
            # Plot all precursors containing the specified metal if no specific precursor is provided 
            else:
                sns.lineplot(
                    data=precursors[(precursors['Metal'] == metal) & (precursors['Measurement'] == 1)], 
                    x='Energy_Corrected', 
                    y='Normalized', 
                    hue='Experiment', 
                    linewidth=3,
                    palette='colorblind'
                    )
            if type(intermediates) != type(None):
                sns.lineplot(
                    data=intermediates[(intermediates['Metal'] == metal) & (intermediates['Measurement'] == 1)], 
                    x='Energy_Corrected', 
                    y='Normalized', 
                    hue='Experiment', 
                    linewidth=3,
                    palette='colorblind'
                    )
            if type(products) != type(None):
                sns.lineplot(
                    data=products[(products['Metal'] == metal) & (products['Measurement'] == 1)], 
                    x='Energy_Corrected', 
                    y='Normalized', 
                    hue='Experiment', 
                    linewidth=3,
                    palette='colorblind'
                    )
        # Plot all measurements of specified metal edge
        sns.lineplot(
            data=data[(data['Metal'] == metal)], 
            x='Energy_Corrected', 
            y='Normalized', 
            hue='Measurement', 
            palette='viridis',
            )
        # Set limits of x-axis to match the edge measurements
        plt.xlim(
            (np.amin(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == np.amin(data['Measurement']))]), 
            np.amax(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == np.amin(data['Measurement']))]))
            )
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Energy [eV]', 
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
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Formatting of the hover label "title"
        x_formatting = '.0f'
        # Plot the measurements of the selected experiment/edge
        fig = px.line(
            data_frame=data[(data['Metal'] == metal)],
            x='Energy_Corrected',
            y='Normalized',
            color='Measurement',
            color_discrete_sequence=px.colors.sample_colorscale('viridis', samplepoints=n_measurements),
        )
        # Change line formatting
        fig.update_traces(
            line=dict(
                width=2,
            ),
            xhoverformat=x_formatting,
        )
        # Plot metal foil if provided
        if type(foils) != type(None):
            # Check if the metal foil exists in the dataset
            assert metal in foils.Metal.unique(), f'No foil with the metal: {metal}\n\nValid values are: {foils.Metal.unique()}'
            foil_filter = (foils['Metal']== metal) & (foils['Measurement'] == 1)
            fig.add_trace(
                go.Scatter(
                    x=foils['Energy_Corrected'][foil_filter], 
                    y=foils['Normalized'][foil_filter], 
                    mode='lines',
                    name=metal+' foil',
                    line=dict(
                        width=3,
                        color='black',
                    ),
                    xhoverformat=x_formatting,
                ))
        # Plot precursor(s) if provided
        if type(precursors) != type(None):
            # Plot only specific precursor if it is provided
            if type(precursor_suffix) != type(None):
                # Check if the specified precursor exists in the dataset
                assert metal+precursor_suffix in precursors.Experiment.unique(), f'No metal with the name: {metal+precursor_suffix}\n\nValid values are: {precursors.Experiment.unique()}'
                precursor_filter = (precursors['Experiment'] == metal+precursor_suffix) & (precursors['Measurement'] == 1)
                fig.add_trace(
                    go.Scatter(
                        x=precursors['Energy_Corrected'][precursor_filter], 
                        y=precursors['Normalized'][precursor_filter], 
                        mode='lines',
                        name=metal+precursor_suffix,
                        line=dict(
                            width=3,
                            color='red',
                        ),
                        xhoverformat=x_formatting,
                    ))
            # Plot all precursors containing the specified metal if no specific precursor is provided 
            else:
                for i, precursor in enumerate(precursors['Experiment'][precursors['Metal'] == metal].unique()):
                    precursor_filter = (precursors['Experiment'] == precursor) & (precursors['Measurement'] == 1)
                    fig.add_trace(
                        go.Scatter(
                            x=precursors['Energy_Corrected'][precursor_filter], 
                            y=precursors['Normalized'][precursor_filter], 
                            mode='lines',
                            name=precursor,
                            line=dict(
                                width=3,
                                color=px.colors.qualitative.D3[i],
                            ),
                            xhoverformat=x_formatting,
                        ))
        if type(products) != type(None):
            for i, product in enumerate(products['Experiment'][products['Metal'] == metal].unique()):
                product_filter = (products['Experiment'] == product) & (products['Measurement'] == 1)
                fig.add_trace(
                    go.Scatter(
                        x=products['Energy_Corrected'][product_filter], 
                        y=products['Normalized'][product_filter], 
                        mode='lines',
                        name=product,
                        line=dict(
                            width=3,
                            color=px.colors.qualitative.D3[i],
                        ),
                        xhoverformat=x_formatting,
                    ))
        if type(intermediates) != type(None):
            for i, intermediate in enumerate(intermediates['Experiment'][intermediates['Metal'] == metal].unique()):
                intermediate_filter = (intermediates['Experiment'] == intermediate) & (intermediates['Measurement'] == 1)
                fig.add_trace(
                    go.Scatter(
                        x=intermediates['Energy_Corrected'][intermediate_filter], 
                        y=intermediates['Normalized'][intermediate_filter], 
                        mode='lines',
                        name=intermediate,
                        line=dict(
                            width=3,
                            color=px.colors.qualitative.D3[i],
                        ),
                        xhoverformat=x_formatting,
                    ))
        # Set limits of x-axis to match the edge measurements
        fig.update_xaxes(
            range=[np.amin(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == np.amin(data['Measurement']))]), 
            np.amax(data['Energy_Corrected'][(data['Metal'] == metal) & (data['Measurement'] == np.amin(data['Measurement']))])]
            )
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Energy [eV]</b>',
            yaxis_title='<b>Normalized</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
        )
        # Customize the hover labels
        hovertemplate = 'Normalized = %{y:.2f}'
        fig.update_traces(hovertemplate=hovertemplate)
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_insitu_waterfall(
    data: pd.DataFrame, 
    experiment: str, 
    lines: Union[list[int], None]=None,
    vmin: Union[float, None]=None,
    vmax: Union[float, None]=None,
    y_axis: str='Measurement',
    time_unit: str='seconds',
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='in-situ_waterfall.png'
) -> None:
    """Waterfall plot of all normalized measurements in an in-situ experiment.

    Reference lines can be added to highlight certain measurements or transitions in the data.

    Args:
        data (pd.DataFrame): The dataset.
        experiment (str): The experiment to plot.
        lines (optional, Union[list[int], None]): The measurment number to draw a horizontal line at. Defaults to None.
        vmin (optional, Union[float, None]): The minimum value in the color range. If "None" the minimum value in the data is used. Defaults to None.
        vmax (optional, Union[float, None]): The maximum value in the color range. If "None" the maximum value in the data is used. Defaults to None.
        y_axis (optional, str): The column to use as the y-axis. Defaults to 'Measurement'.
        time_unit (optional, str): The unit of time to use. Can be either written out (seconds) or just the first letter (s). Defaults to 'seconds'.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'in-situ_waterfall.png'.

    Returns:
        None
    """
    # Check if metal exists in the dataset
    assert experiment in data.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {data.Experiment.unique()}'
    # Collect the relevant columns for plotting
    df_plot = data[['Measurement', 'Relative Time', 'Energy_Corrected', 'Normalized']][data['Experiment'] == experiment]
    # Ensure time unit is all lowercase
    time_unit = time_unit.lower()
    # Set unit specific parameters
    if (time_unit == 'seconds') or (time_unit == 's'):
        # Set time label
        time_label = 'Relative Time [s]'
        # Set conversion value
        unit_conversion = 1
        # Set dtype
        time_dtype = np.int32
        # Set number of decimals
        n_decimals = 0
    elif (time_unit == 'minutes') or (time_unit == 'm'):
        # Set time label
        time_label = 'Relative Time [min]'
        # Set conversion value
        unit_conversion = 60
        # Set dtype
        time_dtype = np.float32
        # Set number of decimals
        n_decimals = 1
    elif (time_unit == 'hours') or (time_unit == 'h'):
        # Set time label
        time_label = 'Relative Time [h]'
        # Set conversion value
        unit_conversion = 60 * 60
        # Set dtype
        time_dtype = np.float32
        # Set number of decimals
        n_decimals = 2
    # Define axis specific variables
    if y_axis == 'Measurement':
        # Set ylabel
        y_label = y_axis
        # Convert y values
        df_plot['Plotting Y'] = df_plot[y_axis]
    elif y_axis == 'Relative Time':
        # Set y_label 
        y_label = time_label
        # Convert y values
        df_plot['Plotting Y'] = (df_plot[y_axis] / unit_conversion).astype(dtype=time_dtype)
    # Create arrays for hover data
    if interactive:
        n_rows = df_plot['Measurement'].unique().shape[0]
        measurement_array = df_plot['Measurement'].to_numpy().reshape(n_rows, -1)
        time_array = (df_plot['Relative Time'] / unit_conversion).astype(dtype=time_dtype).to_numpy().reshape(n_rows, -1)
    # Create pivot table of relevant data
    heatmap_data = df_plot.pivot('Plotting Y', 'Energy_Corrected', 'Normalized')
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(
            data=heatmap_data,
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            cbar=True,
        )
        # Plot horizontial lines
        if lines != None:
            for line in lines:
                position = line - 1 
                plt.axhline(
                    y=position, 
                    color='red', 
                    linestyle='--',
                    linewidth=1.5,
                )
                # Plot annotation text
                plt.annotate(
                    text=f'Measurement {line}',
                    xy=(plt.xlim()[0], position),
                    xytext=(5,3),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='white',
                )
        # Turn on ticks
        plt.tick_params(
            bottom=True,
            left=True,
        )
        # Set xtick labels
        xticks = [
            np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]), 
            np.amax(data['Energy_Corrected'][data['Experiment'] == experiment])
            ] 
        xticks += list(
            np.arange(
                np.ceil(np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
                np.ceil(np.amax(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
                step=50,
                dtype=np.int32,
            )
        )
        plt.xticks(
            ticks=xticks - xticks[0],
            labels=xticks,
            rotation=0,
        )
        # Formatting of y-axis ticks and labels
        # Set ytick positions
        ytick_base = 25
        plt.gca().yaxis.set_major_locator(ticker.IndexLocator(base=ytick_base, offset=ytick_base - 1))
        y_pos = plt.yticks()[0].astype(int)
        y_pos = np.append(y_pos, 0)
        if y_axis == 'Measurement':
            # Set ytick labels
            plt.yticks(
                y_pos,
                data[y_axis][data['Experiment'] == experiment].unique()[y_pos],
            )
        elif y_axis == 'Relative Time':
            # Converted y values
            y_converted = (data[y_axis][data['Experiment'] == experiment].unique() / unit_conversion).astype(dtype=time_dtype)
            # Round values
            y_converted = np.round(
                y_converted,
                decimals=n_decimals,
            )
            # Set ytick labels
            plt.yticks(
                ticks=y_pos,
                labels=y_converted[y_pos],
            )
        # Invert y-axis
        plt.gca().invert_yaxis()
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Energy [eV]', 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            y_label, 
            fontsize=14, 
            fontweight='bold'
            )
        # Format colorbar
        cbar_ax = ax.figure.axes[-1]
        cbar_ax.set_ylabel(
            ylabel='Normalized',
            size=14,
            weight='bold'
        )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Plot the measurements of the selected experiment/edge
        fig = px.imshow(
            img=heatmap_data,
            zmin=vmin,
            zmax=vmax,
            origin='lower',
            color_continuous_scale='viridis',
            aspect='auto',
            labels=dict(color='<b>Normalized</b>'),
        )
        customdata = np.stack((measurement_array, time_array), axis=-1)
        fig.update_traces(
            customdata=customdata,
        )
        # Plot horizontial lines
        if lines != None:
            for line in lines:
                position = line - 1 
                fig.add_hline(
                    y=df_plot['Plotting Y'].unique()[position], 
                    line_color='red', 
                    line_dash='dash',
                    line_width=1.5,
                    annotation_text=f'<b>Measurement {line}</b>',
                    annotation_position='top left',
                    annotation_font=dict(
                        color='white',
                        size=11,
                    )
                )
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Energy [eV]</b>',
            yaxis_title=f'<b>{y_label}</b>',
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
        # Customize the hover labels
        if n_decimals == 0:
            time_format = ' = %{customdata[1]:.0f}<br>'
        elif n_decimals == 1:
            time_format = ' = %{customdata[1]:.1f}<br>'
        elif n_decimals == 2:
            time_format = ' = %{customdata[1]:.2f}<br>'
        hovertemplate = 'Normalized = %{z:.2f}<br>' + 'Measurement = %{customdata[0]:.0f}<br>' + time_label + time_format + 'Energy [eV] = %{x:.0f}<extra></extra>'
        fig.update_traces(hovertemplate=hovertemplate)
        # Add and format spikes
        fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=-2)
        fig.update_yaxes(showspikes=True, spikecolor="red", spikethickness=-2)
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None
# TODO: Make y axis relative to the reference frame
def plot_insitu_change(
    data: pd.DataFrame, 
    experiment: str, 
    reference_measurement: int=1,
    lines: Union[list[int], None]=None,
    vmin: Union[float, None]=None,
    vmax: Union[float, None]=None,
    y_axis: str='Measurement',
    time_unit: str='seconds',
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='in-situ_change.png'
) -> None:
    """Waterfall plot of the difference between all normalized measurements and a reference measurement in an in-situ experiment.

    Reference lines can be added to highlight certain measurements or transitions in the data.

    Args:
        data (pd.DataFrame): The dataset.
        experiment (str): The experiment to plot.
        reference_measurement (int): The measurement number to use as the reference measurement. Defaults to 1.
        lines (optional, Union[list[int], None]): The measurment number to draw a horizontal line at. Defaults to None.
        vmin (optional, Union[float, None]): The minimum value in the color range. If "None" the minimum value in the data is used. Defaults to None.
        vmax (optional, Union[float, None]): The maximum value in the color range. If "None" the maximum value in the data is used. Defaults to None.
        y_axis (optional, str): The column to use as the y-axis. Defaults to 'Measurement'.
        time_unit (optional, str): The unit of time to use. Can be either written out (seconds) or just the first letter (s). Defaults to 'seconds'.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'in-situ_change.png'.

    Returns:
        None
    """
    # Check if metal exists in the dataset
    assert experiment in data.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {data.Experiment.unique()}'
    # Create dataframe from subset of data
    df_change = data[['Measurement', 'Relative Time', 'Energy_Corrected', 'Normalized']][data['Experiment'] == experiment]
    # Extract the reference measurement
    reference_data = df_change['Normalized'][(data['Measurement'] == reference_measurement)].to_numpy()
    # Substract reference from all measurements
    difference_from_reference = df_change['Normalized'].to_numpy().reshape(-1, reference_data.shape[0]) - reference_data
    # Make new column for differences
    df_change['ref_delta'] = difference_from_reference.reshape(-1)
    # Ensure time unit is all lowercase
    time_unit = time_unit.lower()
    # Set unit specific parameters
    if (time_unit == 'seconds') or (time_unit == 's'):
        # Set time label
        time_label = 'Relative Time [s]'
        # Set conversion value
        unit_conversion = 1
        # Set dtype
        time_dtype = np.int32
        # Set number of decimals
        n_decimals = 0
    elif (time_unit == 'minutes') or (time_unit == 'm'):
        # Set time label
        time_label = 'Relative Time [min]'
        # Set conversion value
        unit_conversion = 60
        # Set dtype
        time_dtype = np.float32
        # Set number of decimals
        n_decimals = 1
    elif (time_unit == 'hours') or (time_unit == 'h'):
        # Set time label
        time_label = 'Relative Time [h]'
        # Set conversion value
        unit_conversion = 60 * 60
        # Set dtype
        time_dtype = np.float32
        # Set number of decimals
        n_decimals = 2
    # Define axis specific variables
    if y_axis == 'Measurement':
        # Set ylabel
        y_label = y_axis
        # Convert y values
        df_change['Plotting Y'] = df_change[y_axis]
    elif y_axis == 'Relative Time':
        # Set y_label 
        y_label = time_label
        # Convert y values
        df_change['Plotting Y'] = (df_change[y_axis] / unit_conversion).astype(dtype=time_dtype)
    # Create arrays for hover data
    if interactive:
        n_rows = df_change['Measurement'].unique().shape[0]
        measurement_array = df_change['Measurement'].to_numpy().reshape(n_rows, -1)
        time_array = (df_change['Relative Time'] / unit_conversion).astype(dtype=time_dtype).to_numpy().reshape(n_rows, -1)
    # Create pivot table of relevant data
    heatmap_data = df_change.pivot('Plotting Y', 'Energy_Corrected', 'ref_delta')
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(
            data=heatmap_data,
            center=0.,
            vmin=vmin,
            vmax=vmax,
            cmap='seismic',
            cbar=True,
        )
        # Plot reference line 
        plt.axhline(
            y=reference_measurement - 1, 
            color='k', 
            linestyle='-',
            linewidth=1.5,
        )
        # Plot reference annotation
        plt.annotate(
            text=f'Reference ({reference_measurement})',
            xy=(plt.xlim()[0], reference_measurement-1),
            xytext=(5,3),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='k',
        )
        # Plot horizontial lines
        if lines != None:
            for line in lines:
                position = line - 1
                plt.axhline(
                    y=position, 
                    color='k', 
                    linestyle='--',
                    linewidth=1.5,
                )
                # Plot annotation text
                plt.annotate(
                    text=f'Measurement {line}',
                    xy=(plt.xlim()[0], position),
                    xytext=(5,3),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='k',
                )
        # Turn on ticks
        plt.tick_params(
            bottom=True,
            left=True,
        )
        # Set xtick labels
        xticks = [
            np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]), 
            np.amax(data['Energy_Corrected'][data['Experiment'] == experiment])
            ] 
        xticks += list(
            np.arange(
                np.ceil(np.amin(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
                np.ceil(np.amax(data['Energy_Corrected'][data['Experiment'] == experiment]) / 50) * 50, 
                step=50,
                dtype=np.int32,
            )
        )
        plt.xticks(
            ticks=xticks - xticks[0],
            labels=xticks,
            rotation=0,
        )
        # Formatting of y-axis ticks and labels
        # Set ytick positions
        ytick_base = 25
        plt.gca().yaxis.set_major_locator(ticker.IndexLocator(base=ytick_base, offset=ytick_base - 1))
        y_pos = plt.yticks()[0].astype(int)
        y_pos = np.append(y_pos, 0)
        if y_axis == 'Measurement':
            # Set ytick labels
            plt.yticks(
                y_pos,
                data[y_axis][data['Experiment'] == experiment].unique()[y_pos],
            )
        elif y_axis == 'Relative Time':
            # Converted y values
            y_converted = (data[y_axis][data['Experiment'] == experiment].unique() / unit_conversion).astype(dtype=time_dtype)
            # Round values
            y_converted = np.round(
                y_converted,
                decimals=n_decimals,
            )
            # Set ytick labels
            plt.yticks(
                ticks=y_pos,
                labels=y_converted[y_pos],
            )
        # Invert y-axis
        plt.gca().invert_yaxis()
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Energy [eV]', 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            y_label, 
            fontsize=14, 
            fontweight='bold'
            )
        # Format colorbar
        cbar_ax = ax.figure.axes[-1]
        cbar_ax.set_ylabel(
            ylabel=r'$\mathbf{\Delta}$ Normalized intensity',
            size=14,
            weight='bold'
        )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Plot the measurements of the selected experiment/edge
        fig = px.imshow(
            img=heatmap_data,
            zmin=vmin,
            zmax=vmax,
            origin='lower',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0.,
            aspect='auto',
            labels=dict(color='<b>\u0394 Normalized intensity</b>'),
        )
        customdata = np.stack((measurement_array, time_array), axis=-1)
        fig.update_traces(
            customdata=customdata,
        )
        fig.add_hline(
            y=df_change['Plotting Y'].unique()[reference_measurement - 1], 
            line_color='black', 
            line_width=1.5,
            annotation_text=f'<b>Reference ({reference_measurement})</b>',
            annotation_position='top left',
            annotation_font=dict(
                color='black',
                size=11,
            )
        )
        # Plot horizontial lines
        if lines != None:
            for line in lines:
                position = line - 1 
                fig.add_hline(
                    y=df_change['Plotting Y'].unique()[position], 
                    line_color='black', 
                    line_dash='dash',
                    line_width=1.5,
                    annotation_text=f'<b>Measurement {line}</b>',
                    annotation_position='top left',
                    annotation_font=dict(
                        color='black',
                        size=11,
                    )
                )
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Energy [eV]</b>',
            yaxis_title=f'<b>{y_label}</b>',
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
        # Customize the hover labels
        if n_decimals == 0:
            time_format = ' = %{customdata[1]:.0f}<br>'
        elif n_decimals == 1:
            time_format = ' = %{customdata[1]:.1f}<br>'
        elif n_decimals == 2:
            time_format = ' = %{customdata[1]:.2f}<br>'
        hovertemplate = '\u0394 Normalized intensity = %{z:.2f}<br>' + 'Measurement = %{customdata[0]:.0f}<br>' + time_label + time_format + 'Energy [eV] = %{x:.0f}<extra></extra>'
        fig.update_traces(hovertemplate=hovertemplate)
        # Add and format spikes
        fig.update_xaxes(showspikes=True, spikecolor="black", spikethickness=-2)
        fig.update_yaxes(showspikes=True, spikecolor="black", spikethickness=-2)
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_temperatures(
    df: pd.DataFrame, 
    with_uncertainty: bool=True, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='temperature_curves.png'
) -> None:
    """------------------------------------------------------------------
    Plotting visual comparison of when the different metals in the experiment reduce by showing the weight of the metal foil component determined from Linear Combination Analysis (LCA).

    Args:
        df (pd.DataFrame): Results of LCA.
        with_uncertainty (optional, bool): Boolean flag controlling if the uncertainty on the average emperature is plotted. Defaults to True.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'temperature_curves.png'.

    Returns:
        None
    """  
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(10,8))
        # Plot temperature curves
        sns.lineplot(
            data=df,
            x='Measurement',
            y='Temperature',
            hue='Metal',
            ci=None,
            linewidth=2,
            palette='colorblind',
        )
        # Create filter for relevant values
        avg_filter = (df['Parameter'] == 'product_weight') & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & (df['Metal'] == df['Metal'].unique().tolist()[0])
        # Plot temperature average
        sns.lineplot(
            data=df[avg_filter],
            x='Measurement',
            y='Temperature Average',
            ci=None,
            linewidth=3,
            color='k',
            label='Average',
        )
        # Plot the uncertainty on the average temperature
        if with_uncertainty:
            # Plot uncertainty as the values within +/- 1 standard deviation
            plt.fill_between(
                df['Measurement'][avg_filter], 
                df['Temperature Average'][avg_filter] - df['Temperature Std'][avg_filter], 
                df['Temperature Average'][avg_filter] + df['Temperature Std'][avg_filter], 
                alpha=0.3,
                color='k',
                )
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Measurement', 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            'Temperature [°C]', 
            fontsize=14, 
            fontweight='bold'
            )
        # Specify placement, formatting and title of the legend
        plt.legend(
            loc='center left', 
            bbox_to_anchor=(1,0.5),
            title='Metal',
            fontsize=12,
            title_fontsize=13,
            )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Formatting of the hover label "title"
        x_formatting = '.0f'
        # Create filter for relevant values
        df_filter = (df['Parameter'] == 'product_weight') & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0])
        # Plot the measurements of the selected experiment/edge
        fig = px.line(
            data_frame=df[df_filter],
            x='Measurement',
            y='Temperature',
            color='Metal',
            color_discrete_sequence=sns.color_palette('colorblind').as_hex()
        )
        # Change line formatting
        fig.update_traces(
            line=dict(
                width=2,
            ),
            xhoverformat=x_formatting,
        )
        # Customize the hover labels
        hovertemplate = 'Temperature = %{y:.1f} °C'
        fig.update_traces(hovertemplate=hovertemplate)
        if with_uncertainty:
            # Change hover template for line with uncertainty
            hovertemplate = 'Temperature = %{y:.1f} +/- %{customdata:.1f} °C'
        # Create filter for relevant values
        avg_filter = df_filter & (df['Metal'] == df['Metal'].unique().tolist()[0])
        # Plot temperature average
        fig.add_trace(
            go.Scatter(
                x=df['Measurement'][avg_filter], 
                y=df['Temperature Average'][avg_filter], 
                mode='lines',
                name='Average',
                legendgroup='Average',
                line=dict(
                    width=3,
                    color='black',
                ),
                xhoverformat=x_formatting,
                customdata=df['Temperature Std'][avg_filter],
                hovertemplate=hovertemplate,
            ))
        # Plot the uncertainty on the average temperature
        if with_uncertainty:
            # Create lists with the x-values and upper/lower error bounds
            x_range = df['Measurement'][avg_filter].tolist()
            std_upper = (df['Temperature Average'][avg_filter] + df['Temperature Std'][avg_filter]).tolist()
            std_lower = (df['Temperature Average'][avg_filter] - df['Temperature Std'][avg_filter]).tolist()
            # Plot uncertainty as the values within +/- 1 standard deviation
            fig.add_trace(
                go.Scatter(
                    x=x_range + x_range[::-1], 
                    y=std_upper + std_lower[::-1], 
                    fill='toself',
                    fillcolor='rgba(0,0,0,0.3)',
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup='Average',
                    xhoverformat=x_formatting,
                ))
        # Set limits of x-axis to match the edge measurements
        fig.update_xaxes(
            range=[np.amin(df['Measurement'][avg_filter]), 
            np.amax(df['Measurement'][avg_filter])]
            )
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Measurement</b>',
            yaxis_title='<b>Temperature [°C]</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
        )
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_LCA(
    results: pd.DataFrame, 
    data: pd.DataFrame, 
    experiment: str, 
    measurement: int, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='LCA_plot.png'
) -> None:
    """------------------------------------------------------------------
    Plotting of the result of Linear Combination Analysis (LCA) for a single measurement and combination of metal and precursor.

    Args:
        results (pd.DataFrame): Results from LCA.
        data (pd.DataFrame): Dataset.
        experiment (str): Combination of metal and precursor to plot. Given in the format '{metal} + {precursor}'.
        measurement (int): Measurement to plot.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'LCA_plot.png'.

    Returns:
        None
    """    
    # Check if experiment exists in dataset
    assert experiment in results.Experiment.unique(), f'No experiment with the name: {experiment}\n\nValid values are: {results.Experiment.unique()}'
    # Check if measurement exists in dataset
    assert measurement in results["Measurement"][results["Experiment"] == experiment].unique(), f'No measurement with the name: {measurement}\n\nValid values are: {results["Measurement"][results["Experiment"] == experiment].unique()}'
    # Extract the metal (, intermediate) and precursor from the experiment
    components = experiment.split(' + ')
    if len(components) == 2:
        precursor, product = components
        intermediate = None
    elif len(components) == 3:
        precursor, intermediate, product = components
    # Create filters for relevant values
    data_filter = (data['Experiment'] == data['Experiment'][data['Metal'] == results['Metal'][results['Experiment'] == experiment].unique()[0]].unique()[0]) & (data['Measurement'] == measurement)
    product_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'product_weight') & (results['Measurement'] == measurement)
    precursor_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'precursor_weight') & (results['Measurement'] == measurement)
    if intermediate != None:
        intermediate_filter = (results['Experiment'] == experiment) & (results['Parameter'] == 'intermediate_weight') & (results['Measurement'] == measurement)
    # Scale the basis functions with their component weight
    product_component = (results['Value'][product_filter].to_numpy() * results['Basis Function'][product_filter].to_numpy())[0]
    precursor_component = (results['Value'][precursor_filter].to_numpy() * results['Basis Function'][precursor_filter].to_numpy())[0]
    if intermediate != None:
        intermediate_component = (results['Value'][intermediate_filter].to_numpy() * results['Basis Function'][intermediate_filter].to_numpy())[0]
        component_sum = precursor_component + intermediate_component + product_component
    else:
        component_sum = precursor_component + product_component
    # Define colors to use
    color_list = sns.color_palette('colorblind').as_hex()    
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(8,8))
        # Plot reference line at 0
        plt.axhline(y=0, c='k', alpha=0.5, lw=3, ls='--')
        # Plot measurement
        sns.lineplot(
            data=data[data_filter], 
            x='Energy_Corrected', 
            y='Normalized', 
            color='k',
            linewidth=3,
            label=f'Measurement {measurement}',
            )
        # Plot LCA approximation
        sns.lineplot( 
            x=results['Energy Range'][product_filter].to_numpy()[0], 
            y=component_sum, 
            color='cyan',
            linewidth=3,
            label='LCA approx.'
            )
        # Plot product component
        sns.lineplot( 
            x=results['Energy Range'][product_filter].to_numpy()[0], 
            y=product_component, 
            color=color_list.pop(0),
            linewidth=3,
            label=f'Product ({results["Value"][product_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][product_filter].to_numpy()[0]:.1%})'
            )
        if intermediate != None:
            # Plot intermediate component
            sns.lineplot( 
                x=results['Energy Range'][intermediate_filter].to_numpy()[0], 
                y=intermediate_component, 
                color=color_list.pop(0),
                linewidth=3,
                label=f'Intermediate ({results["Value"][intermediate_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][intermediate_filter].to_numpy()[0]:.1%})'
                )
        # Plot precursor component
        sns.lineplot( 
            x=results['Energy Range'][precursor_filter].to_numpy()[0], 
            y=precursor_component, 
            color=color_list.pop(0),
            linewidth=3,
            label=f'Precursor ({results["Value"][precursor_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][precursor_filter].to_numpy()[0]:.1%})'
            )
        # Plot LCA residual
        sns.lineplot( 
            x=results['Energy Range'][product_filter].to_numpy()[0], 
            y=data['Normalized'][data_filter & data['Energy_Corrected'].isin(results['Energy Range'][product_filter].to_numpy()[0])] - component_sum, 
            color='r',
            linewidth=3,
            label='Residual'
            )
        # Ensure the y-axis covers atleast the range from 0 - 1
        y_lim_bot, y_lim_top = plt.ylim()
        if y_lim_bot > 0:
            y_lim_bot = 0
        if y_lim_top < 1:
            y_lim_top = 1
        # Ensure the y-axis isn't outside -0.5 - 1.5
        if y_lim_bot < -0.5:
            y_lim_bot = -0.5
        if y_lim_top > 1.5:
            y_lim_top = 1.5
        plt.ylim((y_lim_bot, y_lim_top))
        # Set limits of x-axis to match the edge measurements
        plt.xlim(
            (np.amin(data['Energy_Corrected'][data_filter]),
            np.amax(data['Energy_Corrected'][data_filter]))
            )
        # Specify text and formatting of axis labels
        plt.xlabel(
            'Energy [eV]', 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            'Normalized X-ray transmission', 
            fontsize=14, 
            fontweight='bold'
            )
        # Specify placement, formatting and title of the legend
        plt.legend( 
            title=f'{data["Metal"][data_filter].unique()[0]} edge', 
            fontsize=12, 
            title_fontproperties=dict(size=14, weight='bold'),
            ncol=3,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.01)
            )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Formatting of the hover label "title"
        x_formatting = '.0f'
        # Plot the measurements of the selected experiment/edge
        fig = go.Figure(data=go.Scatter(
            x=data['Energy_Corrected'][data_filter],
            y=data['Normalized'][data_filter],
            mode='lines',
            name=f'Measurement {measurement}',
            line=dict(
                width=3,
                color='black',
            ),
            xhoverformat=x_formatting,
        ))
        # Plot reference line at 0
        fig.add_hline(
            y=0,
            line_width=3,
            line_dash='dash',
            line_color='rgba(0,0,0,0.5)'
        )
        # Plot the LCA approximation
        fig.add_trace(
            go.Scatter(
                x=results['Energy Range'][product_filter].to_numpy()[0], 
                y=component_sum, 
                mode='lines',
                name='LCA approx.',
                line=dict(
                    width=3,
                    color='magenta',
                ),
                xhoverformat=x_formatting,
            ))
        # Plot the LCA product component
        fig.add_trace(
            go.Scatter(
                x=results['Energy Range'][product_filter].to_numpy()[0], 
                y=product_component, 
                mode='lines',
                name=f'Product ({results["Value"][product_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][product_filter].to_numpy()[0]:.1%})',
                line=dict(
                    width=3,
                    color=color_list.pop(0),
                ),
                xhoverformat=x_formatting,
            ))
        if intermediate != None:
            # Plot the LCA intermediate component
            fig.add_trace(
                go.Scatter(
                    x=results['Energy Range'][intermediate_filter].to_numpy()[0], 
                    y=intermediate_component, 
                    mode='lines',
                    name=f'Intermediate ({results["Value"][intermediate_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][intermediate_filter].to_numpy()[0]:.1%})',
                    line=dict(
                        width=3,
                        color=color_list.pop(0),
                    ),
                    xhoverformat=x_formatting,
                ))
        # Plot the LCA precursor component
        fig.add_trace(
            go.Scatter(
                x=results['Energy Range'][precursor_filter].to_numpy()[0], 
                y=precursor_component, 
                mode='lines',
                name=f'Precursor ({results["Value"][precursor_filter].to_numpy()[0] * 100:.1f} +/- {results["StdCorrected"][precursor_filter].to_numpy()[0]:.1%})',
                line=dict(
                    width=3,
                    color=color_list.pop(0),
                ),
                xhoverformat=x_formatting,
            ))
        # Plot the LCA residual
        fig.add_trace(
            go.Scatter(
                x=results['Energy Range'][product_filter].to_numpy()[0], 
                y=data['Normalized'][data_filter & data['Energy_Corrected'].isin(results['Energy Range'][product_filter].to_numpy()[0])] - component_sum, 
                mode='lines',
                name='Residual',
                line=dict(
                    width=3,
                    color='red',
                ),
                xhoverformat=x_formatting,
            ))
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title='<b>Energy [eV]</b>',
            yaxis_title='<b>Normalized</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
        )
        # Customize the hover labels
        hovertemplate = 'Normalized = %{y:.2f}'
        fig.update_traces(hovertemplate=hovertemplate)
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_LCA_change(
    df: pd.DataFrame, 
    product: str, 
    precursor: str, 
    intermediate: Union[str, None]=None,
    x_axis: str='Measurement', 
    with_uncertainty: bool=True, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='LCA_change.png'
) -> None:
    """------------------------------------------------------------------
    Plotting the change in weights determined from Linear Combination Analysis (LCA) over an entire experiment.

    Args:
        df (pd.DataFrame): Results from LCA.
        product (str): The product that should be plotted.
        precursor (str): The precursor that should be plotted.
        intermediate (optional, Union[str, None]): The intermediate that should be plotted. Defaults to None.
        x_axis (optional, str): The column to plot on the a-axis. Defaults to 'Measurement'.
        with_uncertainty (optional, bool): Boolean flag controlling if the uncertainties on the weights are plotted. Defaults to True.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'LCA_change.png'.

    Returns:
        None
    """    
    # Check if metal exists in dataset
    assert product in df.Product.unique(), f'No product with the name: {product}\n\nValid values are: {df.Product.unique()}'
    # Check if precursor exists in dataset
    assert precursor in df.Precursor.unique(), f'No precursor with the name: {precursor}\n\nValid values are: {df.Precursor.unique()}'
    # Check if intermediate exists in dataset
    assert intermediate in df.Intermediate.unique(), f'No intermediate with the name: {intermediate}\n\nValid values are: {df.Intermediate.unique()}'
    # Check if x_axis exists in dataset
    assert x_axis in df.columns, f'No column with the name: {x_axis}\n\nValid values are: {df.columns}'
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Define colors to use
        color_list = sns.color_palette('colorblind').as_hex() 
        # Create figure object and set the figure size
        plt.figure(figsize=(10,8))
        # Create filter for relevant values
        df_filter = (df['Product'] == product) & (df['Precursor'] == precursor) 
        if intermediate != None:
            df_filter = df_filter & (df['Intermediate'] == intermediate)
        # Plot weights for the two components for each measurement
        sns.lineplot(
            data=df[df_filter],
            x=x_axis, 
            y='Value', 
            hue='Parameter', 
            ci=None,
            linewidth=3,
            palette='colorblind',
        )
        # TODO: Make proper legend labels
        # Plot the uncertainty on the weights
        if with_uncertainty:
            # Create filter for relevant values
            product_filter = (df['Parameter'] == 'product_weight') & df_filter
            # Plot uncertainty as the values within +/- 1 standard deviation
            plt.fill_between(
                df[x_axis][product_filter], 
                df['Value'][product_filter] - df['StdCorrected'][product_filter], 
                df['Value'][product_filter] + df['StdCorrected'][product_filter], 
                alpha=0.3,
                color=color_list.pop(0),
            )
            if intermediate != None:
                # Create filter for relevant values
                intermediate_filter = (df['Parameter'] == 'intermediate_weight') & df_filter
                # Plot uncertainty as the values within +/- 1 standard deviation
                plt.fill_between(
                    df[x_axis][intermediate_filter], 
                    df['Value'][intermediate_filter] - df['StdCorrected'][intermediate_filter], 
                    df['Value'][intermediate_filter] + df['StdCorrected'][intermediate_filter], 
                    alpha=0.3,
                    color=color_list.pop(0),
                )
            # Create filter for relevant values
            precursor_filter = (df['Parameter'] == 'precursor_weight') & df_filter
            # Plot uncertainty as the values within +/- 1 standard deviation
            plt.fill_between(
                df[x_axis][precursor_filter], 
                df['Value'][precursor_filter] - df['StdCorrected'][precursor_filter], 
                df['Value'][precursor_filter] + df['StdCorrected'][precursor_filter], 
                alpha=0.3,
                color=color_list.pop(0),
            )
        # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
        y_min, y_max = plt.ylim()
        plt.ylim(
            (np.amax([y_min, -0.5]), np.amin([y_max, 1.5]))
        )
        # Specify the units used on the x-axis
        if 'Temperature' in x_axis:
            units = ' [°C]'
        else:
            units = ''
        # Specify text and formatting of axis labels
        plt.xlabel(
            x_axis + units, 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            'Weight fraction', 
            fontsize=14, 
            fontweight='bold'
            )
        # Specify placement, formatting and title of the legend
        labels = [product, precursor]
        if intermediate != None:
            labels = [product, intermediate, precursor]
        plt.legend(
            loc='center left', 
            bbox_to_anchor=(1,0.5),
            labels=labels, 
            title='Components',
            fontsize=12,
            title_fontsize=13,
            )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Define colors to use
        color_list = sns.color_palette('colorblind') 
        # Formatting of the hover label "title"
        x_formatting = '.1f'
        # Variables to catch min and max axis values
        y_min = -0.02
        y_max = 1.02
        # Create filter for relevant values
        df_filter = (df['Product'] == product) & (df['Precursor'] == precursor) 
        if intermediate != None:
            df_filter = df_filter & (df['Intermediate'] == intermediate)
        # Plot the LCA weights over time
        fig = px.line(
            data_frame=df[df_filter],
            x=x_axis,
            y='Value',
            color='Parameter',
            color_discrete_sequence=color_list.as_hex(),
            custom_data=['StdCorrected'],
        )
        # Change line formatting
        fig.update_traces(
            line=dict(
                width=3,
            ),
            xhoverformat=x_formatting,
        )
        if with_uncertainty:
            # Customize the hover labels
            hovertemplate = 'Weight = %{y:.2f} +/- %{customdata[0]:.2f}'
            fig.update_traces(hovertemplate=hovertemplate)
        else:
            # Customize the hover labels
            hovertemplate = 'Weight = %{y:.2f}'
            fig.update_traces(hovertemplate=hovertemplate)
        # Plot the uncertainty on the weights
        if with_uncertainty:
            # Create filter for relevant values
            product_filter = (df['Parameter'] == 'product_weight') & df_filter
            # Create lists with the x-values and upper/lower error bounds
            x_range = df[x_axis][product_filter].tolist()
            std_upper = (df['Value'][product_filter] + df['StdCorrected'][product_filter]).tolist()
            std_lower = (df['Value'][product_filter] - df['StdCorrected'][product_filter]).tolist()
            # Plot uncertainty as the values within +/- 1 standard deviation
            fig.add_trace(
                go.Scatter(
                    x=x_range + x_range[::-1], 
                    y=std_upper + std_lower[::-1], 
                    fill='toself',
                    fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup='product_weight',
                    xhoverformat=x_formatting,
                ))
            if intermediate != None:
                # Create filter for relevant values
                intermediate_filter = (df['Parameter'] == 'intermediate_weight') & df_filter
                # Create lists with the x-values and upper/lower error bounds
                x_range = df[x_axis][intermediate_filter].tolist()
                std_upper = (df['Value'][intermediate_filter] + df['StdCorrected'][intermediate_filter]).tolist()
                std_lower = (df['Value'][intermediate_filter] - df['StdCorrected'][intermediate_filter]).tolist()
                # Plot uncertainty as the values within +/- 1 standard deviation
                fig.add_trace(
                    go.Scatter(
                        x=x_range + x_range[::-1], 
                        y=std_upper + std_lower[::-1], 
                        fill='toself',
                        fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
                        line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup='intermediate_weight',
                        xhoverformat=x_formatting,
                    ))  
            # Create filter for relevant values
            precursor_filter = (df['Parameter'] == 'precursor_weight') & df_filter
            # Create lists with the x-values and upper/lower error bounds
            x_range = df[x_axis][precursor_filter].tolist()
            std_upper = (df['Value'][precursor_filter] + df['StdCorrected'][precursor_filter]).tolist()
            std_lower = (df['Value'][precursor_filter] - df['StdCorrected'][precursor_filter]).tolist()
            # Plot uncertainty as the values within +/- 1 standard deviation
            fig.add_trace(
                go.Scatter(
                    x=x_range + x_range[::-1], 
                    y=std_upper + std_lower[::-1], 
                    fill='toself',
                    fillcolor=f'rgba{(*color_list.pop(0), 0.3)}',
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup='precursor_weight',
                    xhoverformat=x_formatting,
                ))  
            # Check if largest value is above current axis range
            if y_max < np.amax(std_upper):
                y_max = np.amax(std_upper)
            # Check if smallest values is below current axis range
            if y_min > np.amin(std_lower):
                y_min = np.amin(std_lower)
        # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
        fig.update_yaxes(
            range=[np.amax([y_min, -0.5]), np.amin([y_max, 1.5])]
        )
        # Set limits of x-axis to match the edge measurements
        fig.update_xaxes(
            range=[np.amin(df[x_axis][df_filter]), 
            np.amax(df[x_axis][df_filter])]
            )
        # Specify the units used on the x-axis
        if 'Temperature' in x_axis:
            units = ' [°C]'
        else:
            units = ''
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title=f'<b>{x_axis}{units}</b>',
            yaxis_title='<b>Weight fraction</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
        )
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None

def plot_reduction_comparison(
    df: pd.DataFrame, 
    precursor_type: str='all', 
    x_axis: str='Measurement', 
    with_uncertainty: bool=True, 
    interactive: bool=False, 
    save_plot: bool=False, 
    save_name: str='reduction_comparison.png'
) -> None:
    """------------------------------------------------------------------
    Plotting visual comparison of when the different metals in the experiment reduce by showing the weight of the metal foil component determined from Linear Combination Analysis (LCA).

    Args:
        df (pd.DataFrame): Results of LCA.
        precursor_type (optional, str): The type of precursors to be plotted. Defaults to 'all'.
        x_axis (optional, str): The column to plot on the a-axis. Defaults to 'Measurement'.
        with_uncertainty (optional, bool): Boolean flag controlling if the uncertainties on the weights are plotted. Defaults to True.
        interactive (optional, bool): Boolean flag controlling if the plot is interactive. Defaults to False.
        save_plot (optional, bool): Boolean flag controlling if the plot is saved. Defaults to False.
        save_name (optional, str): The filename of the saved plot. Defaults to 'reduction_comparison.png'.

    Returns:
        None
    """    
    # Check if x_axis exists in dataset
    assert x_axis in df.columns, f'No column with the name: {x_axis}\n\nValid values are: {df.columns}'
    # Make figure using matplotlib and seaborn
    if not interactive:
        # Create figure object and set the figure size
        plt.figure(figsize=(10,8))
        # Plot the weight of the foil component for all metal + precursor combinations
        if precursor_type == 'all':
            df_filter = (df['Parameter'] == 'product_weight')
            sns.lineplot(
                data=df[df_filter], 
                x=x_axis,
                y='Value',
                hue='Metal',
                style='Precursor Type',
                ci=None,
                linewidth=2,
                palette='colorblind',
                )
        # Plot the weight of the foil component for all experiments with the specified type of precursor
        else:
            # Check if the precursor type exists in the dataset
            assert precursor_type in df['Precursor Type'].unique(), f'No precursor type with the name: {precursor_type}\n\nValid values are: {df["Precursor Type"].unique()}'
            # Create filter for relevant values
            df_filter = (df['Parameter'] == 'product_weight') & (df['Precursor'].str.contains(precursor_type))
            sns.lineplot(
                data=df[df_filter], 
                x=x_axis,
                y='Value',
                hue='Metal',
                ci=None,
                linewidth=2,
                palette='colorblind',
                )
        # Plot uncertainties on the foil weights
        if with_uncertainty:
            # Loop over each metal
            for i, metal in enumerate(df['Metal'][df_filter].unique()):
                # If all metal + precursor combinations are plotted, only show uncertainties for the most common precursor type to avoid visual clutter
                if precursor_type == 'all':
                    # Create filter for relevant values
                    foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & df_filter
                # If a precursor type is specified, only plot uncertainties for that type of precursor
                else:
                    # Create filter for relevant values
                    foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type) & df_filter
                # Plot uncertainties
                plt.fill_between(
                    df[x_axis][foil_filter], 
                    df['Value'][foil_filter] - df['StdCorrected'][foil_filter], 
                    df['Value'][foil_filter] + df['StdCorrected'][foil_filter], 
                    alpha=0.3,
                    color=sns.color_palette('colorblind').as_hex()[i],
                    )
        # Ensure the y-axis covers atleast the range from 0 - 1
        y_lim_bot, y_lim_top = plt.ylim()
        if y_lim_bot > 0:
            y_lim_bot = 0
        if y_lim_top < 1:
            y_lim_top = 1
        # Ensure the y-axis isn't outside -0.5 - 1.5
        if y_lim_bot < -0.5:
            y_lim_bot = -0.5
        if y_lim_top > 1.5:
            y_lim_top = 1.5
        plt.ylim((y_lim_bot, y_lim_top))
        # Specify the units used on the x-axis
        if 'Temperature' in x_axis:
            units = ' [°C]'
        else:
            units = ''
        # Specify text and formatting of axis labels
        plt.xlabel(
            x_axis + units, 
            fontsize=14, 
            fontweight='bold'
            )
        plt.ylabel(
            'Weight fraction', 
            fontsize=14, 
            fontweight='bold'
            )
        # Specify placement, formatting and title of the legend
        plt.legend(
            loc='center left', 
            bbox_to_anchor=(1,0.5),
            labels=df['Experiment'][df_filter].unique(), 
            title='Components',
            fontsize=12,
            title_fontsize=13,
            )
        # Enforce matplotlibs tight layout
        plt.tight_layout()
        # Save plot as a png
        if save_plot:
            plt.savefig(f'./Data/Plots/{save_name}', dpi=300)
        plt.show()
    # Make interactive plot using plotly
    elif interactive:
        # Formatting of the hover label "title"
        x_formatting = '.1f'
        # Variables to catch min and max axis values
        y_min = -0.02
        y_max = 1.02
        fig = go.Figure()
        # Plot the weight of the foil component for all metal + precursor combinations
        if precursor_type == 'all':
            df_filter = (df['Parameter'] == 'product_weight')
            # Plot the LCA weights over time
            for i, metal in enumerate(df['Metal'][df_filter].unique()):
                # Loop over each relevant precursor for each metal
                for precursor_type in df['Precursor Type'][df_filter & (df['Metal'] == metal)].unique():
                    # Create filter for relevant values
                    foil_filter = df_filter & (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type)
                    # Determine linestyle based on most common precursor type
                    if precursor_type == df['Precursor Type'].mode().to_list()[0]:
                        linestyle = 'solid'
                        # Used to link line and the corresponding uncertainty
                        legend_group = f'{metal}'
                    else:
                        linestyle = 'dash'
                        legend_group = None
                    # Plot the trace
                    fig.add_trace(
                        go.Scatter(
                            x=df[x_axis][foil_filter], 
                            y=df['Value'][foil_filter], 
                            mode='lines',
                            name=f'{metal} + {precursor_type}',
                            line_color=f'rgba{sns.color_palette("colorblind")[i]}',
                            legendgroup=legend_group,
                            line_dash=linestyle,
                            line_width=2,
                            customdata=df['StdCorrected'][foil_filter],
                            xhoverformat=x_formatting,
                        ))
        # Plot the weight of the foil component for all experiments with the specified type of precursor
        else:
            # Check if the precursor type exists in the dataset
            assert precursor_type in df['Precursor Type'].unique(), f'No precursor type with the name: {precursor_type}\n\nValid values are: {df["Precursor Type"].unique()}'
            # Create filter for relevant values
            df_filter = (df['Parameter'] == 'product_weight') & (df['Precursor'].str.contains(precursor_type))
            # Plot the LCA weights over time
            for i, metal in enumerate(df['Metal'][df_filter].unique()):
                # Create filter for relevant values
                foil_filter = df_filter & (df['Metal'] == metal)
                linestyle = 'solid'
                # Used to link line and the corresponding uncertainty
                legend_group = f'{metal}'
                # Plot the trace
                fig.add_trace(
                    go.Scatter(
                        x=df[x_axis][foil_filter], 
                        y=df['Value'][foil_filter], 
                        mode='lines',
                        name=f'{metal} + {precursor_type}',
                        line_color=f'rgba{sns.color_palette("colorblind")[i]}',
                        legendgroup=legend_group,
                        line_dash=linestyle,
                        line_width=2,
                        customdata=df['StdCorrected'][foil_filter],
                        xhoverformat=x_formatting,
                    ))
        if with_uncertainty:
            # Customize the hover labels
            hovertemplate = 'Weight = %{y:.2f} +/- %{customdata:.2f}'
            fig.update_traces(hovertemplate=hovertemplate)
        else:
            # Customize the hover labels
            hovertemplate = 'Weight = %{y:.2f}'
            fig.update_traces(hovertemplate=hovertemplate)
        # Plot uncertainties on the foil weights
        if with_uncertainty:
            # Loop over each metal
            for i, metal in enumerate(df['Metal'][df_filter].unique()):
                # If all metal + precursor combinations are plotted, only show uncertainties for the most common precursor type to avoid visual clutter
                if precursor_type == 'all':
                    # Create filter for relevant values
                    foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == df['Precursor Type'].mode().to_list()[0]) & df_filter
                # If a precursor type is specified, only plot uncertainties for that type of precursor
                else:
                    # Create filter for relevant values
                    foil_filter = (df['Metal'] == metal) & (df['Precursor Type'] == precursor_type) & df_filter
                # Create lists with the x-values and upper/lower error bounds
                x_range = df[x_axis][foil_filter].tolist()
                std_upper = (df['Value'][foil_filter] + df['StdCorrected'][foil_filter]).tolist()
                std_lower = (df['Value'][foil_filter] - df['StdCorrected'][foil_filter]).tolist()
                # Check if largest value is above current axis range
                if y_max < np.amax(std_upper):
                    y_max = np.amax(std_upper)
                # Check if smallest values is below current axis range
                if y_min > np.amin(std_lower):
                    y_min = np.amin(std_lower)
                # Plot uncertainty as the values within +/- 1 standard deviation
                fig.add_trace(
                    go.Scatter(
                        x=x_range + x_range[::-1], 
                        y=std_upper + std_lower[::-1], 
                        fill='toself',
                        fillcolor=f'rgba{(*sns.color_palette("colorblind")[i], 0.3)}',
                        line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f'{metal}',
                        xhoverformat=x_formatting,
                    ))
        # Ensure the y-axis covers atleast the range from 0 - 1 and is not outside -0.5 - 1.5
        fig.update_yaxes(
            range=[np.amax([y_min, -0.5]), np.amin([y_max, 1.5])]
        )
        # Set limits of x-axis to match the edge measurements
        fig.update_xaxes(
            range=[np.amin(df[x_axis][df_filter]), 
            np.amax(df[x_axis][df_filter])]
            )
        # Specify the units used on the x-axis
        if 'Temperature' in x_axis:
            units = ' [°C]'
        else:
            units = ''
        # Specify text and formatting of axis labels
        fig.update_layout(
            xaxis_title=f'<b>{x_axis}{units}</b>',
            yaxis_title='<b>Weight fraction</b>',
            font=dict(
                size=14,
            ),
            hovermode='x unified',
            legend_title='Experiment',
        )
        # Save plot as an image
        if save_plot:
            fig.write_image(f'./Data/Plots/{save_name}')
        fig.show()
    return None