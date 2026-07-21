import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from meteostat import Stations, Hourly
from scipy.stats import spearmanr 
from typing import Callable, List, Dict
import warnings
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
data_folder = DATASETS / "Seifrid_group_weather_data"
from matplotlib import rcParams

# Set global font sizes
rcParams.update({
    'axes.titlesize': 16,       # Title font size
    'axes.labelsize': 14,       # X and Y label font size
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,      # X tick label font size
    'ytick.labelsize': 12,      # Y tick label font size
    'legend.fontsize': 12,      # Legend font size
    'font.size': 12             # Default font size
})



def add_outdoor_temperature_and_relative_humidity(weather_data, location_coordinates, date_col='date_time', radius=28000):
    """Add outdoor temperature and relative humidity from Meteostat hourly data."""
    weather_data = weather_data.copy()
    weather_data['date_time_utc'] = pd.to_datetime(weather_data[date_col], utc=True, errors='coerce')
    weather_data['rounded_datetime'] = weather_data['date_time_utc'].dt.round('h').dt.tz_convert(None)

    lat, lon = location_coordinates
    stations = Stations().nearby(lat, lon, radius=radius)
    stations = stations.fetch(5)

    if stations.empty:
        raise ValueError("No Meteostat stations were found near the selected location.")

    station_ids = stations.index.tolist()
    distances = stations['distance'].replace(0, 1)
    weights = 1 / distances
    weights = weights / weights.sum()

    start = weather_data['rounded_datetime'].min().to_pydatetime()
    end = weather_data['rounded_datetime'].max().to_pydatetime()
    hourly_data_by_station = []
    for station_id in station_ids:
        station_hourly_data = Hourly(station_id, start=start, end=end).fetch()
        if not station_hourly_data.empty:
            station_hourly_data = station_hourly_data.copy()
            station_hourly_data.index.name = 'time'
            station_hourly_data['station'] = station_id
            hourly_data_by_station.append(station_hourly_data)

    if not hourly_data_by_station:
        raise ValueError("No Meteostat hourly data were returned for the selected date range.")

    hourly_data = pd.concat(hourly_data_by_station).reset_index()
    hourly_data['weight'] = hourly_data['station'].map(weights)

    valid_temp = hourly_data['temp'].notna()
    valid_rh = hourly_data['rhum'].notna()

    hourly_data.loc[valid_temp, 'weighted_temp'] = (
        hourly_data.loc[valid_temp, 'temp'] * hourly_data.loc[valid_temp, 'weight']
    )
    hourly_data.loc[valid_rh, 'weighted_rh'] = (
        hourly_data.loc[valid_rh, 'rhum'] * hourly_data.loc[valid_rh, 'weight']
    )

    temp_data = hourly_data.loc[valid_temp].groupby('time').agg(
        weighted_temp=('weighted_temp', 'sum'),
        temp_weight=('weight', 'sum')
    )
    rh_data = hourly_data.loc[valid_rh].groupby('time').agg(
        weighted_rh=('weighted_rh', 'sum'),
        rh_weight=('weight', 'sum')
    )

    outdoor_weather = temp_data.join(rh_data, how='outer')
    outdoor_weather['outdoor temperature'] = outdoor_weather['weighted_temp'] / outdoor_weather['temp_weight']
    outdoor_weather['outdoor relative humidity'] = outdoor_weather['weighted_rh'] / outdoor_weather['rh_weight']
    outdoor_weather = outdoor_weather[['outdoor temperature', 'outdoor relative humidity']]
    outdoor_weather.index.name = 'rounded_datetime'

    weather_data = weather_data.merge(
        outdoor_weather,
        left_on='rounded_datetime',
        right_index=True,
        how='left'
    )

    return weather_data.drop(columns=['date_time_utc', 'rounded_datetime'])


def plot_indoor_outdoor_temperature_and_humidity(
        weather_data, save_folder, date_col='date_time', resample_rule='1h',
        show_raw_points=False, font_size=14):
    """Plot indoor/outdoor temperature and humidity after resampling dense sensor data."""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    indoor_color = '#fa6e81'
    outdoor_color = '#41557d'
    tick_font_size = font_size - 2

    temp_cols = ['SHT31 Temperature', 'outdoor temperature']
    humidity_cols = ['SHT31 Relative Humidity', 'outdoor relative humidity']
    plot_cols = temp_cols + humidity_cols

    plot_data = weather_data.copy()
    plot_data['datetime'] = pd.to_datetime(plot_data[date_col], utc=True, errors='coerce').dt.tz_convert(None)
    plot_data = plot_data.dropna(subset=['datetime']).sort_values('datetime')

    for col in plot_cols:
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')

    resampled_data = (
        plot_data
        .set_index('datetime')[plot_cols]
        .resample(resample_rule)
        .median()
    )

    smooth_data = resampled_data.rolling(24, min_periods=1).mean()

    temp_pho = spearmanr(
        plot_data['SHT31 Temperature'],
        plot_data['outdoor temperature'],
        nan_policy='omit'
    )
    humidity_pho = spearmanr(
        plot_data['SHT31 Relative Humidity'],
        plot_data['outdoor relative humidity'],
        nan_policy='omit'
    )

    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    if show_raw_points:
        axes[0].scatter(plot_data['datetime'], plot_data['SHT31 Temperature'],
                        s=6, alpha=0.08, color=indoor_color)
        axes[0].scatter(plot_data['datetime'], plot_data['outdoor temperature'],
                        s=6, alpha=0.08, color=outdoor_color)
        axes[1].scatter(plot_data['datetime'], plot_data['SHT31 Relative Humidity'],
                        s=6, alpha=0.08, color=indoor_color)
        axes[1].scatter(plot_data['datetime'], plot_data['outdoor relative humidity'],
                        s=6, alpha=0.08, color=outdoor_color)

    axes[0].plot(resampled_data.index, resampled_data['SHT31 Temperature'],
                 color=indoor_color, alpha=0.35, linewidth=1)
    axes[0].plot(resampled_data.index, resampled_data['outdoor temperature'],
                 color=outdoor_color, alpha=0.35, linewidth=1)
    indoor_line, = axes[0].plot(smooth_data.index, smooth_data['SHT31 Temperature'],
                                color=indoor_color, linewidth=2.5, label='Indoor')
    outdoor_line, = axes[0].plot(smooth_data.index, smooth_data['outdoor temperature'],
                                 color=outdoor_color, linewidth=2.5, label='Outdoor')
    axes[0].set_ylabel('Temperature (C)', fontsize=font_size)
    axes[0].text(
        0.02, 0.95,
        f"Spearman r = {temp_pho.statistic:.3f}",
        transform=axes[0].transAxes,
        va='top',
        ha='left',
        fontsize=tick_font_size,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray',boxstyle="round,pad=0.3")
    )

    axes[1].plot(resampled_data.index, resampled_data['SHT31 Relative Humidity'],
                 color=indoor_color, alpha=0.35, linewidth=1)
    axes[1].plot(resampled_data.index, resampled_data['outdoor relative humidity'],
                 color=outdoor_color, alpha=0.35, linewidth=1)
    axes[1].plot(smooth_data.index, smooth_data['SHT31 Relative Humidity'],
                 color=indoor_color, linewidth=2.5)
    axes[1].plot(smooth_data.index, smooth_data['outdoor relative humidity'],
                 color=outdoor_color, linewidth=2.5)
    axes[1].set_ylabel('Relative Humidity (%)', fontsize=font_size)
    axes[1].set_xlabel('Date', fontsize=font_size)
    axes[1].text(
        0.02, 0.95,
        f"Spearman r = {humidity_pho.statistic:.3f}",
        transform=axes[1].transAxes,
        va='top',
        ha='left',
        fontsize=tick_font_size,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray',boxstyle="round,pad=0.3")
    )

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='both', labelsize=tick_font_size)

    fig.legend(
        handles=[indoor_line, outdoor_line],
        labels=['Indoor', 'Outdoor'],
        loc='upper center',
        ncol=2,
        frameon=True,
        fontsize=font_size
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.autofmt_xdate()

    output_png = save_folder / f"indoor_outdoor_temperature_humidity_{resample_rule}.png"
    # output_pdf = save_folder / f"indoor_outdoor_temperature_humidity_{resample_rule}.pdf"
    fig.savefig(output_png, dpi=900, bbox_inches='tight')
    # fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":

    ### calculating outdoor temperature and relative humidity from Meteostat data for the Seifrid group weather data#
    # location_coordinates = (35.771317, -78.675484)

    # input_file = data_folder / "pth_data_2025-11-06T16-53_to_2026-07-16T16-53.csv"
    # output_file = data_folder / "pth_data_2025-11-06T16-53_to_2026-07-16T16-53_with_outdoor_weather.csv"

    # weather_data = pd.read_csv(input_file)
    # weather_data = add_outdoor_temperature_and_relative_humidity(weather_data, location_coordinates)
    # weather_data.to_csv(output_file, index=False)

    # print(f"Saved weather data with outdoor Meteostat columns to: {output_file}")
    
    #### illustration ####
    
    weather_data = pd.read_csv(data_folder / "pth_data_2025-11-06T16-53_to_2026-07-16T16-53_with_outdoor_weather.csv")
    figure_folder = HERE / "figures"
    plot_indoor_outdoor_temperature_and_humidity(
        weather_data,
        save_folder=figure_folder,
        resample_rule='1h',
        show_raw_points=False,
        font_size=18
    )

    print("Saved plots and resampled data:")

    #### Spearman correlation between indoor and outdoor temperature and relative humidity ###
    temp_pho = spearmanr(weather_data['SHT31 Temperature'], weather_data['outdoor temperature'], nan_policy='omit')
    humidity_pho = spearmanr(weather_data['SHT31 Relative Humidity'], weather_data['outdoor relative humidity'], nan_policy='omit')

    print(f"Spearman correlation between indoor and outdoor temperature: {temp_pho.statistic:.4f}, p-value: {temp_pho.pvalue:.4e}")
    print(f"Spearman correlation between indoor and outdoor relative humidity: {humidity_pho.statistic:.4f}, p-value: {humidity_pho.pvalue:.4e}")
