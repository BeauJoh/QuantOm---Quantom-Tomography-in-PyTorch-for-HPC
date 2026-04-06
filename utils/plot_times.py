#!/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from natsort import humansorted
from os import getenv
from sys import argv

time_file_path  = getenv('TIME_FILE_PATH')
figure_path     = getenv('FIGURE_PATH')
title           = getenv('TITLE')
if not time_file_path: time_file_path = "./times.csv"
if not figure_path: figure_path = "implementation-performance-comparison.pdf"
if not title: title = "Performance"
time_files = glob.glob(time_file_path)

implementations = argv[1:]

df = pd.DataFrame()
for time_file in time_files:
    times = pd.read_csv(time_file,index_col=0)
    df = pd.concat([df,times])
df = df[df.implementation.isin(implementations)] # Apply filter to only show specified implementations

#convert to millions of events
df.events = (df.events/10**6).astype(int)

fig, ax = plt.subplots(figsize=(10,7.5))
lp = sns.lineplot(data=df, x="events", y="duration", hue="implementation")
plt.title(title,fontsize=18)
#sort the legend labels
handles, labels = lp.get_legend_handles_labels()
labels, handles = zip(*humansorted(zip(labels, handles)))
plt.legend(handles, labels, title='Implementation',title_fontsize=15)
plt.tick_params(labelsize=12)
lp.set_ylabel("Average Time (s)",fontsize=15)
lp.set_xlabel("# Events (Millions)",fontsize=15)
fig = lp.get_figure()
fig.savefig(figure_path) 

