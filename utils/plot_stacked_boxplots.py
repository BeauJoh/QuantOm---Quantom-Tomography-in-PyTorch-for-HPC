#!/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from natsort import humansorted
from os import getenv

plt_title   = getenv('TITLE')
result_path = getenv('RESULT_PATH')
figure_path = getenv('FIGURE_PATH')
impl_subs   = getenv("IMPLEMENTATION_SUBSTITUTIONS")
scale_axes  = getenv("SCALE_AXES")
drop_legend = getenv("DROP_LEGEND")
drop_y_ax   = getenv("DROP_Y_AX")
if not plt_title: plt_title = "Scaling and Breakdown of 2D-LOITS"
if not result_path: sycl_result_path = "./scaling-results/*.csv"
if not figure_path: figure_path = "stacked_barplot.pdf"
if not impl_subs: impl_subs = ""

time_files = glob.glob(result_path)

from sys import argv
implementations = argv[1:]

#collate
df = pd.DataFrame()
for time_file in time_files:
    tdf = pd.read_csv(time_file,names=['region','secs'])
    time_file = time_file.replace('.csv','')
    events = int(time_file.split('_').pop())
    tdf = tdf.assign(events=events)
    impl = time_file.split('_')[0].split('/').pop()
    tdf = tdf.assign(implementation=impl)
    #drop the first run (burner)
    tdf = tdf.iloc[1:].reset_index(drop=True)
    df = pd.concat([df,tdf])
df['events'] = df['events'].astype(int) # recast threads from string to int
df = df.sort_values('events') # and sort the final data-frame
df['events'] = df['events'].astype(str) # recast threads back to string to avoid seaborn cleverness
for i in impl_subs.split(","):
    old,new = i.split(":")
    df = df.replace(old,new)
df = df[df.implementation.isin(implementations)] # Apply filter to only show specified implementations
# df must have: region, secs, events, implementation
# (your sample already matches this)

#reduce to just means
full_df = df.groupby(["events","implementation", "region"], as_index=False)["secs"].mean()
# find unique events groups
event_groups = sorted(df["events"].unique(), key=int)
n = len(event_groups)
if n == 1:
    axes = [axes]
#fig, axes = plt.subplots(n, 1, figsize=(12, 4*n), sharex=True)
fig, axes = plt.subplots(n, 1, figsize=(4.135,11.69), sharex=True)
handle,labels=None,None
for ax, ev in zip(axes, event_groups):
    df = full_df[full_df["events"]==ev]

    # 1. Separate out total rows
    totals = df[df["region"] == "total"].set_index("implementation")["secs"]
    df_regions = df[df["region"] != "total"].copy()
    df_regions = df_regions[df_regions["region"] != "forward_single_sample"].copy()

    df_plot = df[df["region"] != "total"].copy()
    df_plot = df_plot[df_plot["region"] != "forward_single_sample"].copy()
    
    region_impl = df_plot.groupby(["implementation", "region"])["secs"].mean().unstack(fill_value=0)

    # 5. Sort regions by total time (optional → prettier ordering)
    #region_impl = region_impl.loc[:, region_impl.mean().sort_values(ascending=False).index]
    # 5. Sort regions alphabetically
    region_impl = region_impl.reindex(sorted(region_impl.columns),axis=1)

    # high-contrast, colorblind-friendly
    palette_colors = sns.color_palette("colorblind", n_colors=len(region_impl.columns))
    # map region name -> color
    region_colors = dict(zip(region_impl.columns, palette_colors))

    # 6. Plot stacked bars
    #fig, ax = plt.subplots(figsize=(10,7.5))
    bottom = None
    for region in region_impl.columns:
        ax.bar(
            region_impl.index,
            region_impl[region],
            bottom=bottom,
            color=region_colors[region],
            label=region
        )
        if bottom is None:
            bottom = region_impl[region].copy()
        else:
            bottom += region_impl[region]
    # Labels / style
    #ax.set_ylabel("Time (seconds)",fontsize=18)
    ax.set_title("at {} events".format(ev))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45,ha="right")
    plt.tick_params(labelsize=12)

    if labels is None:
        handle, labels = ax.get_legend_handles_labels()
if not drop_y_ax:
    fig.supylabel("Time (seconds)", fontsize=18)
#plt.suptitle(plt_title,fontsize=21)
plt.tight_layout()
plt.savefig(figure_path)

if not drop_legend:
    #plt.legend(handle, labels, loc='upper left',title="Region", bbox_to_anchor=(1.01, 1.01))
    ax_prev = ax
    fig, ax = plt.subplots(figsize=(2, 4))
    ax.axis("off")

    handles, labels = ax_prev.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center",
        title="Region"
    )
    fig.savefig(figure_path.replace('.pdf','_legend.pdf'), bbox_inches="tight")

