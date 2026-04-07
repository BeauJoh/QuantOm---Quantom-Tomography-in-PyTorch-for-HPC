import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from plotting_tools import plot_clustered_stacked


def parse_out(df: pd.DataFrame, bottom_indexes: dict[str, str]) -> dict[str, float]:
    data = {}
    for idx, val in df.groupby("component"):
        if idx in bottom_indexes:
            data[idx] = val[2:].time.mean()
    for i in set(bottom_indexes).difference(df.component.unique()):
        data[i] = 0.0
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)

    # Keep hatch styling even on matplotlib builds that do not expose
    # Rectangle.set_hatch_linewidth(...)
    mpl.rcParams["hatch.linewidth"] = 0.15

    spacing = 0.05
    width = 0.2
    bottom_indexes = {
        "pybind": "PyBind Overhead",
        "calc_grid_indices": "Calc. Grid Indicies",
        "calc_weight_matrix": "Calc. Weight Matrix",
        "calc_rho": "Calc. Rho",
        "calc_cdf": "Calc. CDF",
        "uobs_fill_and_flattern": "Fill and Flatten",
        "bin_obs_flattern": "Flatten",
        "linear_interpolation": "Linear Interpolation",
        "reshape": "Reshape",
        "filter": "Filter",
    }
    xticks_conv = {
        "dpcp-tbb": "DPC++\n(TBB)",
        "acpp-omp": "ACPP\n(OpenMP)",
        "cpp": "C++",
        "omp": "C++\n(OpenMP)",
        "pytorch-omp": "PyTorch\n(OpenMP)",
    }

    # Top panel timing-breakdown stems do not exactly match the bottom-panel
    # FRS CSV stems for PyTorch CPU in this artifact.
    file_stem_conv = {
        "pytorch-omp": "pytorch-cpu",
        "acpp-omp": "acpp-omp",
        "dpcp-tbb": "dpcp-tbb",
        "omp": "omp",
        "cpp": "cpp",
    }

    group_order = {
        10000: "",
        100000: "////",
        1000000: "+++",
        10000000: "..",
    }

    rows = []
    for filename in glob.glob(str(results_root / "scaling-results-cpu" / "*.csv")):
        stem = Path(filename).stem
        x, n = stem.split("_")
        n = int(n)
        libdevice = x

        df0 = pd.read_csv(filename, names=["component", "time"])
        data = parse_out(df0, bottom_indexes)
        avg_df = pd.Series(data)

        pct_df = (avg_df / avg_df.sum()) * 100
        rows.append(pd.Series(pct_df.to_dict() | {"N": n, "libdevice": libdevice}))

    if not rows:
        raise RuntimeError(f"No CPU scaling breakdown CSV files found under: {results_root / 'scaling-results-cpu'}")

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(2, sharex="col", figsize=(12, 6), dpi=200)
    out = plot_clustered_stacked(
        df,
        ax[0],
        "libdevice",
        df.libdevice.unique(),
        "N",
        group_order,
        title=None,
        xlim=(-0.7, 5.3),
        width=width,
        spacing=spacing,
    )
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("% of Runtime Spent\nin Component")

    ticks = ax[1].get_xticks()
    labels = ax[1].get_xticklabels()

    for idx, i in enumerate(labels):
        stem = file_stem_conv.get(i.get_text(), i.get_text())

        for jdx, n in enumerate(group_order.keys()):
            csv_path = results_root / "frs_scaling_cpu_results" / f"{stem}_{n}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CPU FRS CSV: {csv_path}")

            df0 = pd.read_csv(csv_path).duration[2:]
            mt = df0.mean().item()
            st = df0.std().item()

            x = ticks[idx] + (width + spacing) * (jdx - 2) + (spacing + width) / 2
            ax[1].bar(
                x,
                mt,
                width=width,
                color="grey",
                yerr=st,
                hatch=group_order[n],
            )

    ax[1].set_yscale("log")
    ax[1].tick_params(axis="x", rotation=30, labelsize=10)
    ax[1].set_ylabel("Execution Time(s)")

    fig.subplots_adjust(wspace=0.10, hspace=0.12, top=0.96, bottom=0.12, right=0.85, left=0.07)
    ax[0].legend(
        out[0].get_legend_handles_labels()[0][0:10],
        [bottom_indexes[i] for i in df.columns[:-2]],
        fontsize=8,
        bbox_to_anchor=(1.1, -0.1),
        loc="lower center",
        ncols=1,
    )

    ticks = [xticks_conv[l.get_text()] for l in ax[1].get_xticklabels()]
    ax[1].set_xticklabels(ticks)
    ax[1].legend(
        out[1][0],
        [f"{i} Events" for i in group_order.keys()],
        fontsize=8,
        bbox_to_anchor=(1.1, 0.4),
        loc="lower center",
        ncols=1,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
