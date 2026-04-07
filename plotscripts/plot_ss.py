import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
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

    result_glob = str(Path(args.results_root) / "strong_scaling_results" / "*.csv")
    df = []

    for filename in glob.glob(result_glob):
        stem = Path(filename).stem
        n = stem.split("_")[-1]
        name = "-".join(stem.split("_")[:-1])

        duration = pd.read_csv(filename).duration
        t = duration[2:]
        df.append({"libdevice": name, "N": int(n), "t": t})

    df = pd.DataFrame(df)
    cpp_t = df[(df.libdevice == "cpp") & (df.N == 1)].t.to_numpy()[0].mean().item()
    df = df[df.libdevice != "cpp"]

    df["mt"] = list(map(lambda x: x.mean(), (cpp_t / df.t)))
    df["st"] = list(map(lambda x: x.std(), (cpp_t / df.t)))

    m = df.pivot(index="N", columns="libdevice", values="mt").sort_index()
    e = df.pivot(index="N", columns="libdevice", values="st").reindex(index=m.index, columns=m.columns)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    m.plot(kind="bar", yerr=e, ax=ax, legend=None, edgecolor="black", capsize=2, width=0.7)
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Speedup over Serial C++")
    line = ax.axhline(1.0, color="black", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.10, hspace=0.12, top=0.96, bottom=0.12, right=0.97, left=0.11)
    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles, [XTICKS_CONV[i] for i in labels], fontsize=9, bbox_to_anchor=(0.0, 1.0), loc="upper left", ncols=2)
    leg2 = ax.legend([line], ["Serial C++ Baseline"], bbox_to_anchor=(0.5, 1.0), loc="upper left")
    ax.add_artist(leg1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
