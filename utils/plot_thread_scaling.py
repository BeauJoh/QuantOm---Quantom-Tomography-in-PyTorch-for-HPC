#!/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from natsort import humansorted
from os import getenv

plt_title = getenv('TITLE')
omp_result_path = getenv('OMP_RESULT_PATH')
sycl_result_path = getenv('SYCL_RESULT_PATH')
py_result_path  = getenv('PY_RESULT_PATH')
extra_result_path  = getenv('EXTRA_RESULT_PATH')
figure_path     = getenv('FIGURE_PATH')
if not plt_title: plt_title = "Core Performance"
if not sycl_result_path: sycl_result_path = "./sycl/*.csv"
if not omp_result_path: omp_result_path = "./omp/*.csv"
if not py_result_path: py_result_path   = "./py/*.csv"
if not figure_path: figure_path = "scaling.pdf"

time_files = glob.glob(omp_result_path)
time_files += glob.glob(sycl_result_path)
time_files += glob.glob(py_result_path)
if extra_result_path:
    time_files += glob.glob(extra_result_path)

from sys import argv
regions = argv[1:]

#collate
df = pd.DataFrame()
for time_file in time_files:
    tdf = pd.read_csv(time_file,header=None)
    tdf = tdf.rename(columns={0:'region',1:'secs'})
    time_file = time_file.replace('.csv','')
    threads = time_file.split('_')[1]
    tdf = tdf.assign(threads=threads)
    df = pd.concat([df,tdf])
df = df[df.region.isin(regions)] # Apply filter to only show specified ROIs
df['threads'] = df['threads'].astype(int) # recast threads from string to int
df = df.sort_values('threads') # and sort the final data-frame
df['threads'] = df['threads'].astype(str) # recast threads back to string to avoid seaborn cleverness
#plot
fig, ax = plt.subplots(figsize=(10,7.5))
#get x-axis order 
lp = sns.lineplot(data=df, x="threads", y="secs", hue="region")
#sort the legend labels
handles, labels = lp.get_legend_handles_labels()
labels, handles = zip(*humansorted(zip(labels, handles)))
#plot it
plt.legend(handles, labels, title='Region',title_fontsize=18)
plt.title(plt_title,fontsize=21)
plt.tick_params(labelsize=12)
lp.set_ylabel("Average Time [s]",fontsize=18)
lp.set_xlabel("# of Threads",fontsize=18)
fig = lp.get_figure()
fig.savefig(figure_path)
