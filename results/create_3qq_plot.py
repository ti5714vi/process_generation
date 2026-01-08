
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.optimize import curve_fit
import math
import copy 
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 14})

# range of multiplicity
x = [2,3,4,5,6]

with open("results.dat") as f:
    lines = f.readlines()[3:]  # skip first 3 lines
with open("results_3qq.dat") as f:
    lines2 = f.readlines()[3:]  # skip first 3 lines

timings=[]
this_time=[]
errors = np.empty(5, dtype=object)
this_error=[]

timings2=[]
this_time2=[]
errors2 = np.empty(5, dtype=object)
this_error2=[]

xsecs=[]
this_xsec=[]

xsecs2=[]
this_xsec2=[]

colors = [to_rgb('tab:blue'), to_rgb('tab:orange'), to_rgb('tab:green'), to_rgb('tab:red')]
shade1 = [tuple([c * 0.5 for c in colors[l]]) for l in range(0,len(colors))]
shade2 = [tuple([c+(1-c) * 0.7 for c in colors[l]]) for l in range(0,len(colors))]

idx=0
for line in lines:
    if line.strip() == "":  # empty row signals end of block
        if this_time:
            timings.append(np.array(this_time, dtype=float))
            this_time = []
            errors[idx]=np.array(this_error, dtype=float)
            idx=idx+1
            this_error = []
            xsecs.append(np.array(this_xsec, dtype=float))
            this_xsec = []
    else:
        this_time.append(float(line.split()[5]))
        this_error.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])
        this_xsec.append(float(line.split()[1]))
idx=0
for line in lines2:
    if line.strip() == "":  # empty row signals end of block
        if this_time2:
            timings2.append(np.array(this_time2, dtype=float))
            this_time2 = []
            errors2[idx]=np.array(this_error2, dtype=float)
            idx=idx+1
            this_error2 = []
            xsecs2.append(np.array(this_xsec2, dtype=float))
            this_xsec2= []
    else:
        this_time2.append(float(line.split()[5]))
        this_error2.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])
        this_xsec2.append(float(line.split()[1]))

# Add last block if file does not end with empty line
if this_time:
    timings.append(np.array(this_time, dtype=float))
if this_error:
    errors[idx]=np.array(this_error, dtype=float)
if this_xsec:
    xsecs.append(np.array(this_xsec, dtype=float))

# latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# for shared axes
fig, axs = plt.subplots(
    2, 2,
    sharex=True,
    sharey=False,
    figsize=(6, 4),
    constrained_layout=False
)

# plot
num_subplots = 4  # 2x2
s = np.empty(4)
for idx in range(num_subplots):
    i = idx // 2  # row
    j = idx % 2   # column
    ax = axs[i, j]
    ax.set_xticks([2, 3, 4, 5, 6])
    if (idx == 0): tag=0
    if (idx == 1): tag=1
    if (idx == 2): tag=0
    if (idx == 3): tag=1
    if (idx == 0 or idx == 1):
        ax.errorbar(x, timings[tag], yerr=errors[tag].T, marker='o', color=colors[idx],linestyle='', markersize=2,capsize=3, label="Total time")
        ax.set_yscale('log')
        ax.tick_params(direction="in")
        ax.errorbar(x, timings2[tag], yerr=errors2[tag].T, marker='o', color=shade1[idx],linestyle='', markersize=2,capsize=3, label="Total time")
    elif (idx == 2 or idx == 3):
        ax.tick_params(direction="in")
        ax.set_yscale('linear')
        ax.plot(x, xsecs[tag]/xsecs2[tag], marker='o', color=shade1[idx],linestyle='', markersize=2,  label="Ratio")

handles, labels = plt.gca().get_legend_handles_labels()

# labels on each plot
axs[0,0].text(0.1, 0.85, "$p p \\rightarrow nj$", transform=axs[0,0].transAxes)
axs[0,1].text(0.1, 0.85, "$p p \\rightarrow t \\bar{t} + (n-2)j$", transform=axs[0,1].transAxes)

upper_left = axs[0,0].get_position()
lower_left = axs[0,1].get_position()

axs[0,0].set_xlabel(r"$n$")
axs[0,1].set_xlabel(r"$n$")
axs[0,0].set_ylabel(r"time $t$ [s]")
axs[1,0].set_ylabel(r"$\frac{\sigma_{\mathrm{3qq}}}{\sigma_{\mathrm{no-3qq}}}$")

# Improve layout
fig.tight_layout()
# Remove spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0.1)

# Optional: show ticks only on outer axes
for ax in axs.flat:
    ax.label_outer()

# Save the plot as a PDF  
plt.savefig("3qq_plot.pdf", format="pdf")  

