import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

XTICKS_CONV = {
    "dpcp-tbb": "DPC++\n(TBB)",
    "acpp-omp": "ACPP\n(OpenMP)",
    "cpp": "C++",
    "omp": "C++\n(OpenMP)",
    "pytorch": "PyTorch\n(OpenMP)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_glob = str(Path(args.results_root) / "ws_scaling_cpu_results" / "*.csv")
    rows = []

    for filename in glob.glob(result_glob):
        stem = Path(filename).stem.removesuffix("cores")
        name, n = stem.split("_")

        duration = pd.read_csv(filename).duration
        samples = duration[2:]  # skip warmup rows

        if len(samples) == 0:
            raise RuntimeError(f"{filename} has no valid samples after warmup removal")

        mt = float(samples.mean())

        # keep real error bars when available; only fall back to 0 if impossible
        if len(samples) < 2:
            st = 0.0
        else:
            st = float(samples.std())
            if not np.isfinite(st):
                st = 0.0

        rows.append(
            {
                "libdevice": name,
                "N": int(n),
                "mt": mt,
                "st": st,
                "nsamples": len(samples),
            }
        )

    if not rows:
        raise RuntimeError(f"No weak scaling CSV files found matching: {result_glob}")

    df = pd.DataFrame(rows)
    print(df[["libdevice", "N", "nsamples", "st"]].sort_values(["libdevice", "N"]).to_string(index=False))

    expected_impls = {"pytorch", "cpp", "omp", "acpp-omp", "dpcp-tbb"}
    expected_cores = {1, 2, 4, 8, 16, 32, 64}

    found_impls = set(df["libdevice"].unique())
    found_cores = set(df["N"].unique())

    missing_impls = expected_impls - found_impls
    missing_cores = expected_cores - found_cores

    if missing_impls:
        raise RuntimeError(f"Missing implementations in weak scaling results: {sorted(missing_impls)}")
    if missing_cores:
        raise RuntimeError(f"Missing core counts in weak scaling results: {sorted(missing_cores)}")

    missing_pairs = []
    for impl in sorted(expected_impls):
        for n in sorted(expected_cores):
            subset = df[(df.libdevice == impl) & (df.N == n)]
            if subset.empty:
                missing_pairs.append((impl, n))

    if missing_pairs:
        raise RuntimeError(f"Missing weak scaling result rows for: {missing_pairs}")

    order = ["acpp-omp", "dpcp-tbb", "omp", "pytorch", "cpp"]

    m = df.pivot(index="N", columns="libdevice", values="mt").sort_index()[order]
    e = (
        df.pivot(index="N", columns="libdevice", values="st")
        .reindex(index=m.index, columns=m.columns)[order]
        .fillna(0.0)
    )

    if m.isna().any().any():
        bad = m[m.isna()]
        raise RuntimeError(f"Weak scaling mean matrix contains NaNs:\n{bad}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    x = np.arange(len(m.index))
    ncols = len(m.columns)
    total_width = 0.8
    bar_width = total_width / ncols

    for j, col in enumerate(m.columns):
        xpos = x - total_width / 2 + j * bar_width + bar_width / 2
        y = m[col].to_numpy(dtype=float)
        yerr = e[col].to_numpy(dtype=float)

        if not np.isfinite(y).all():
            raise RuntimeError(f"Non-finite means for {col}: {y}")
        if not np.isfinite(yerr).all():
            raise RuntimeError(f"Non-finite std-devs for {col}: {yerr}")

        ax.bar(
            xpos,
            y,
            width=bar_width,
            yerr=yerr,
            label=col,
            edgecolor="black",
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in m.index])
    ax.set_yscale("log")
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Execution Time(s)")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.10, hspace=0.12, top=0.96, bottom=0.12, right=0.95, left=0.10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [XTICKS_CONV[i] for i in labels],
        fontsize=9,
        bbox_to_anchor=(0.0, 1.0),
        loc="upper left",
        ncols=2,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
