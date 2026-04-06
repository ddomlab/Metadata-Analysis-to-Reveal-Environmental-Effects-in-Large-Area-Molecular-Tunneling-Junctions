
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
from scipy.stats import spearmanr


def distribution_plot(
    df: pd.DataFrame,
    parameters: List[str],
    log_cutoff: int = 100,
    columns_per_row: int = 2,
    figsize=(16, 10.5),
    title: str = "Distribution Plots",
    epsilon=1e-6,
    fontsize: int = 14,
    force_all_ticks: List[str] = None,
):
    if force_all_ticks is None:
        force_all_ticks = []

    num_plots = len(parameters)
    num_rows = (num_plots - 1) // columns_per_row + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=columns_per_row, figsize=figsize)
    fig.suptitle(title, fontsize=int(fontsize * 1.4), fontweight='bold')

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(parameters):
        row_idx = i // columns_per_row
        col_idx = i % columns_per_row
        ax = axes[row_idx, col_idx]

        data = df[column]
        data_range = data.max() / (data.min() + epsilon)
        log_scale = (data_range >= log_cutoff and "test" not in column)

        # ---------------------------------------------------------
        # SPECIAL HANDLING FOR DISCRETE FEATURES WITH FULL TICKS
        # ---------------------------------------------------------
        if column in force_all_ticks:
            unique_vals = np.sort(data.dropna().unique())

            # Build centered bins: e.g., 0->[-0.5, 0.5], 1->[0.5,1.5]
            bins = np.concatenate((
                [unique_vals[0] - 0.5],
                (unique_vals[:-1] + unique_vals[1:]) / 2,
                [unique_vals[-1] + 0.5]
            ))

            sns.histplot(
                data,
                ax=ax,
                bins=bins,
                color='#259AC1',
                kde=False
            )

            # TICKS IN THE CENTER OF BINS
            ax.set_xticks(unique_vals)
            ax.set_xticklabels(unique_vals, fontsize=int(fontsize * 0.9), rotation=0)

        else:
            # Normal histogram
            sns.histplot(
                data,
                ax=ax,
                kde=False,
                color='#259AC1',
                log_scale=log_scale
            )

        # Labels & formatting
        ax.set_xlabel(column, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel("Count", fontweight='bold', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=int(fontsize * 0.9))

        # Inset boxplot
        inset = ax.inset_axes([0.01, -0.667, 0.99, 0.2])
        sns.boxplot(
            x=data,
            ax=inset,
            color="#259AC1",
            log_scale=log_scale if column not in force_all_ticks else False
        )
        inset.set(yticks=[], xlabel=None)
        inset.tick_params(axis='x', labelsize=int(fontsize * 0.8))

    # Clean unused axes
    for j in range(i + 1, num_rows * columns_per_row):
        r = j // columns_per_row
        c = j % columns_per_row
        fig.delaxes(axes[r, c])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()




## Correlation + Distribution + scattering Plots in one heat pair plot

def corrfunc(x, y, **kwds):
    cmap = kwds['cmap']
    norm = kwds['norm']
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)

    # Calculate Spearman correlation
    r, _ = spearmanr(x, y)  # Change to spearmanr
    facecolor = cmap(norm(r))
    ax.set_facecolor(facecolor)

    lightness = (max(facecolor[:3]) + min(facecolor[:3])) / 2
    ax.annotate(f"{r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                color='white' if lightness < 0.7 else 'black', size=40, ha='center', va='center')

# Borrowed from
#   - http://stackoverflow.com/a/31385996/4099925
#   - https://stackoverflow.com/questions/31385375/seaborn-pairwise-matrix-of-hexbin-jointplots/31385996#31385996
def plot_hexbin(x, y, color="blue",
                # max_series=None, min_series=None,
                **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    # xmin, xmax = min_series[x.name], max_series[x.name]
    # ymin, ymax = min_series[y.name], max_series[y.name]
    plt.hexbin(x, y, gridsize=12, cmap=cmap,
              #  extent=[xmin, xmax, ymin, ymax],
               **kwargs
               )


def draw_heat_pair_plot(df: pd.DataFrame, parameters: list, title: str = '', hex_bin=False):

    m_df = df[parameters]
    # Create a PairGrid
    g = sns.PairGrid(m_df)
    g.fig.set_dpi(200)
    if hex_bin:
      g.map_lower(plot_hexbin, color='blue')
    else:
      g.map_lower(plt.scatter, s=10)
    g.map_diag(sns.histplot, kde=False)
    g.map_upper(corrfunc, cmap=plt.get_cmap("vlag"), norm=plt.Normalize(vmin=-1, vmax=1))

    # Adjust ticks and labels for size
    for ax in g.axes.flatten():
        ax.tick_params(axis='both', labelsize=20)  # Increase tick size
        # ax.set_xlabel(ax.get_xlabel(), fontsize=16, rotation=45)  # Increase x-label size
        # ax.set_ylabel(ax.get_ylabel(), fontsize=16)  # Increase y-label size


        ax.set_xlabel(ax.get_xlabel(), fontsize=28, fontweight='bold',rotation=65)  # Set x label font
        ax.set_ylabel(ax.get_ylabel(), fontsize=35, fontweight='bold',rotation=0)  # Set y label font
        ax.yaxis.get_label().set_horizontalalignment('right')

    # Set the figure title (optional)
    g.fig.suptitle(title, fontsize=60, y=1.02)

    g.fig.subplots_adjust(wspace=0.1, hspace=0.06)  # Equal spacing in both directions
    plt.show()