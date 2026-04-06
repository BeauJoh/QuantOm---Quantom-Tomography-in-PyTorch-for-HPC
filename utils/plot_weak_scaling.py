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
scale_axes  = getenv("SCALE_AXES")
if not plt_title: plt_title = "Core Performance"
if not result_path: sycl_result_path = "./weak_scaling_results/*.csv"
if not figure_path: figure_path = "weak_scaling.pdf"
if not impl_subs: impl_subs = ""

time_files = glob.glob(result_path)

from sys import argv
implementations = argv[1:]

#collate
df = pd.DataFrame()
for time_file in time_files:
    #print(time_file)
    #import ipdb; ipdb.set_trace()
    tdf = pd.read_csv(time_file)
    time_file = time_file.replace('.csv','')
    time_file = time_file.split('/').pop()
    impl = time_file.split('_')[0]
    ncores = time_file.split('_')[1]
    ncores = ncores.replace('cores','')
    ncores = int(ncores)
    tdf = tdf.assign(ncores=ncores,implementation=impl)
    df = pd.concat([df,tdf])

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

df = df.drop(df.columns[[0]],axis=1)

#plot
fig, ax = plt.subplots(figsize=(10,7.5))

#get x-axis order 
lp = sns.lineplot(data=df, x="ncores", y="duration", hue="implementation", hue_order=present_impls, palette=impl_colors)
#sort the legend labels
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*humansorted(zip(labels, handles)))
#plot it
plt.legend(handles, labels, title='Implementation',title_fontsize=18)
#plt.title(plt_title,fontsize=21)
plt.tick_params(labelsize=12)
threads = sorted(df["ncores"].unique())
ticks = 2 ** np.arange(int(np.log2(min(threads))), int(np.log2(max(threads))) + 1)
plt.xticks(ticks)  # place ticks at powers of 2
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
lp.set_ylabel("Time (seconds)",fontsize=18)
lp.set_xlabel("Number of Cores",fontsize=18)

if scale_axes:
    plt.yscale("log")
fig = lp.get_figure()
fig.savefig(figure_path)
