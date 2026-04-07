import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colormaps as cm
from typing import Dict, List, Tuple, Any


def plot_clustered_stacked(
    df: pd.DataFrame,
    ax: Axes,
    xtick_name: str,
    xticks: List[str],
    group_name: str,
    group_order: Dict[str, str],
    title: str = "multiple stacked bar plot",
    width: float = 0.2,
    xlim: Tuple[float, float] = (-0.1, 5.5),
    spacing: float = 0.05,
    padding: float = 0.15,
    hatch_lw: float = 0.1,
    widths = None,
    colors=None,
    bottom_indexes=None,
    **kwargs: Any,
) -> Tuple[Axes, Tuple[List[Any], Dict[str, str]]]:
    """
    Plot a clustered stacked bar chart with hatching for group differentiation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the values to plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object to draw on.
    xtick_name : str
        Column name used for x-axis tick grouping.
    xticks : list[str]
        Ordered list of x-axis tick labels to display.
    group_name : str
        Column name used to group the data.
    group_order : dict[str, str]
        Mapping of group names to hatch patterns.
    score1_ids : list[str]
        Columns from df to include in the stacked bars.
    title : str, default="multiple stacked bar plot"
        Title of the chart.
    width : float, default=0.2
        Width of each bar.
    xlim : tuple[float, float], default=(-0.1, 5.5)
        Limits for the x-axis.
    spacing : float, default=0.05
        Spacing between groups within a cluster.
    padding : float, default=0.15
        Padding between clusters of xticks.
    hatch_lw : float, default=0.1
        Line width of hatch patterns.
    **kwargs : Any
        Extra keyword arguments passed to pandas plot.

    Returns
    -------
    axe : matplotlib.axes.Axes
        The modified matplotlib Axes with the plot.
    legend_info : tuple[list, dict[str, str]]
        Tuple of dummy legend handles and the group_order mapping.
    """
    dfall = []
    dfg = df.groupby(group_name)
    nvalues = len(df.columns) - 2
    for mode in group_order.keys():
        group = dfg.get_group(mode)
        temp = group.set_index(xtick_name)
        dfall.append(temp.loc[[i for i in xticks if i in temp.index]])

    t_lens = []
    modes = []
    for i in xticks:
        t = pd.DataFrame(
            [k.loc[i] for k in dfall if i in k.index]
        ).set_index(group_name)
        if len(t) > 0:
            bottom = np.zeros(len(t))
            for idx, col in enumerate(t.columns):
                if colors:
                    color = colors[idx]
                else:
                    color = cm.get('tab10')(idx)

                p = ax.bar(
                    t.index,
                    t[col],
                    bottom=t[bottom_indexes[col]].sum(axis=1) if bottom_indexes else bottom,
                    label=idx,
                    color=color,
                    alpha=kwargs["alpha"][idx] if "alpha" in kwargs else 1.0,
                )
                bottom += t[col].values

        t_lens.append(len(t))
        modes.append(t.index.map(lambda x: group_order[x]).to_list())
    # When nvalues > xticks, this is needed for it go not out of range
    for i in range(nvalues - len(xticks)):
        modes.append(t.index.map(lambda x: group_order[x]).to_list())

    t_lens = np.array(t_lens)
    h, _ = ax.get_legend_handles_labels()
    indexes = np.zeros(len(t_lens))
    width_spaced = width + spacing
    for i in range(1, len(t_lens)):
        indexes[i] = (
            width_spaced * (t_lens[i - 1] / 2.0 + t_lens[i] / 2.0) + indexes[i - 1] + padding
        )
    for jdx, i in enumerate(range(0, len(xticks) * nvalues, nvalues)):
        for idx, pa in enumerate(h[i : i + nvalues]):
            for kdx, rect in enumerate(pa.patches):
                if widths:
                    rect.set_width(widths[idx])
                else:
                    rect.set_width(width)
                rect.set_hatch(modes[idx][kdx])
                rect.set_hatch_linewidth(hatch_lw)
                rect.set_x(
                    kdx * width_spaced
                    - len(pa.patches) * width_spaced / 2.0
                    + spacing / 2.0
                    + indexes[jdx]
                )

    ax.set_xticks(indexes, xticks, rotation=30)
    ax.set_xlim(*xlim)
    ax.set_title(title)

    n = []
    for i in group_order.keys():
        n.append(
            ax.bar(
                0, 0, color="gray", hatch=group_order[i] * 2, hatch_linewidth=hatch_lw
            )
        )

    return ax, (n, group_order)
