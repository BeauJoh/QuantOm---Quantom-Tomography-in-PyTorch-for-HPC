#!/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from natsort import humansorted
from os import getenv
from matplotlib.ticker import FuncFormatter

plt_title   = getenv('TITLE')
result_path = getenv('RESULT_PATH')
figure_path = getenv('FIGURE_PATH')
impl_subs   = getenv("IMPLEMENTATION_SUBSTITUTIONS")
use_speedup = getenv("USE_SPEEDUP")
show_ideal_scaling=getenv("SHOW_IDEAL_SCALING")
if not plt_title: plt_title = "Core Performance"
if not result_path: sycl_result_path = "./strong_scaling_results/*.csv"
if not figure_path: figure_path = "strong_scaling.pdf"
if not impl_subs: impl_subs = ""

time_files = glob.glob(result_path)

from sys import argv
implementations = argv[1:]

#collate
df = pd.DataFrame()
for time_file in time_files:
    tdf = pd.read_csv(time_file)
    #tdf = tdf.rename(columns={0:'region',1:'secs'})
    time_file = time_file.replace('.csv','')
    threads = int(time_file.split('_').pop())
    tdf = tdf.assign(threads=threads)
    #drop the first run (burner)
    tdf = tdf.iloc[1:].reset_index(drop=True)
    df = pd.concat([df,tdf])
df = df.drop(df.columns[[0]],axis=1)
df['threads'] = df['threads'].astype(int) # recast threads from string to int
df = df.sort_values('threads') # and sort the final data-frame
df['threads'] = df['threads'].astype(str) # recast threads back to string to avoid seaborn cleverness
for i in impl_subs.split(","):
    old,new = i.split(":")
    df = df.replace(old,new)
print(df.implementation.unique())
df = df[df.implementation.isin(implementations)] # Apply filter to only show specified implementations

all_impls = [
    "C++", "OpenMP",
    "PyTorch (CPU)", "PyTorch (CUDA)", "PyTorch (HIP)", "PyTorch (XPU)",
    "SYCL AdaptiveCpp (OpenMP)", "SYCL AdaptiveCpp (CUDA)", "SYCL AdaptiveCpp (HIP)",
    "SYCL DPC++ (TBB)", "SYCL DPC++ (CUDA)", "SYCL DPC++ (HIP)", "SYCL DPC++ (XPU)"
]
# Pick a high-contrast, colorblind-friendly palette
# We'll use tab20 (up to 20 colors) since we have 13 categories
palette_colors = sns.color_palette("tab20", n_colors=len(all_impls))
# Static lookup: implementation -> color
impl_colors = dict(zip(all_impls, palette_colors))
present_impls = [impl for impl in all_impls if impl in df["implementation"].unique()]

if use_speedup:

    df["threads"] = df["threads"].astype(int)
    # get baseline runtimes (1 thread)
    baselines = (
        df[df["threads"] == 1]
        .groupby(["host", "events", "implementation"])["duration"]
        .mean()
    )

    # compute strong-scaling speedup
    df["speedup"] = df.apply(lambda row: baselines.loc[
        (row["host"], row["events"], row["implementation"])
    ] / row["duration"],
    axis=1)

    # compute speedup = baseline / duration
    #df["speedup"] = df["baseline"] / df["duration"]
    #plot
    fig, ax = plt.subplots(figsize=(10,7.5))
    #get x-axis order 
    lp = sns.lineplot(data=df, x="threads", y="speedup", hue="implementation", hue_order=present_impls, palette=impl_colors)
    #sort the legend labels
    handles, labels = lp.get_legend_handles_labels()
    labels, handles = zip(*humansorted(zip(labels, handles)))
    #plot it
    plt.legend(handles, labels, title='Implementation',title_fontsize=18)
    #plt.title(plt_title,fontsize=21)
    plt.tick_params(labelsize=12)

    threads = sorted(df["threads"].unique())
    ticks = 2 ** np.arange(int(np.log2(min(threads))), int(np.log2(max(threads))) + 1)
    plt.xticks(ticks)  # place ticks at powers of 2
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    if show_ideal_scaling:
        threads = sorted(df["threads"].unique())
        plt.plot(threads, threads, "--", color="gray", label="Ideal")
    lp.set_ylabel("Speedup",fontsize=18)
else:
    #plot
    fig, ax = plt.subplots(figsize=(10,7.5))
    #get x-axis order 
    lp = sns.lineplot(data=df, x="threads", y="duration", hue="implementation", hue_order=present_impls, palette=impl_colors)
    #sort the legend labels
    handles, labels = lp.get_legend_handles_labels()
    labels, handles = zip(*humansorted(zip(labels, handles)))
    #plot it
    plt.legend(handles, labels, title='Implementation',title_fontsize=18)
    #plt.title(plt_title,fontsize=21)
    plt.tick_params(labelsize=12)
    lp.set_ylabel("Time (seconds)",fontsize=18)
lp.set_xlabel("Number of Threads",fontsize=18)
fig = lp.get_figure()
fig.savefig(figure_path)
