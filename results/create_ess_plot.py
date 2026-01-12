
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.optimize import curve_fit
import math
import copy 
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 16})

# range of multiplicity
x = [2,3,4,5,6]

with open("results.dat") as f:
    lines = f.readlines()[3:]  # skip first 3 lines

timings=[]
this_time=[]
errors = []
errors = np.empty(5, dtype=object)
this_error=[]
rwgts=[]
this_rwgt=[]
totals=[]
this_tot=[]

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
            rwgts.append(np.array(this_rwgt, dtype=float))
            this_rwgt = []
            totals.append(np.array(this_tot, dtype=float))
            this_tot = []
    else:
        this_time.append(float(line.split()[28]))
        this_error.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])
        this_rwgt.append(float(line.split()[25]))

# Add last block if file does not end with empty line
if this_time:
    timings.append(np.array(this_time, dtype=float))
if this_error:
    errors[idx]=np.array(this_error, dtype=float)
if this_rwgt:
    rwgts.append(np.array(this_rwgt, dtype=float))
if this_time:
    totals.append(np.array(this_tot, dtype=float))

# latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# for shared axes
fig, axs = plt.subplots(
    1, 2,
    sharex=True,
    sharey=False,
    figsize=(12, 6),
    constrained_layout=False
)


labels = ["$p p \\rightarrow nj$",
          "$p p \\rightarrow t \\bar{t} + (n-2)j$",
          "$p p \\rightarrow ZZ + (n-2)j$",
          "$p p \\rightarrow e^+e^- + (n-2)j$"]

# plot
num_values=[0,1,2,3]
ax=axs   

for pl in range(0,2):
    ax=axs[pl]
    for nr in num_values:
        if (nr == 0): tag=0
        if (nr == 1): tag=2
        if (nr == 2): tag=3
        if (nr == 3): tag=1
        if (pl == 1):
            ax.plot(x, timings[tag]/100000 , marker='o', color=colors[tag],linestyle='--', 
                  markersize=4, linewidth=0.4,label=labels[tag])
        if (pl == 0):
            ax.plot(x, rwgts[tag] , marker='o', color=colors[tag],linestyle='--',
                  markersize=4, linewidth=0.4,label=labels[tag])

axs[0].set_xlabel(r"$n$")
axs[1].set_xlabel(r"$n$")
axs[1].set_ylabel(r"$f_{\mathrm{ESS}}$")
axs[0].set_ylabel(r"$u^{\mathrm{eff}}$")
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
fig = plt.gcf()
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 0, 1]

# add common legend
axs[1].legend(loc="lower left",
    ncol=1)

# Improve layout
fig.tight_layout()
# Remove spacing between subplots
#plt.subplots_adjust(wspace=0, hspace=0)

# Save the plot as a PDF  
plt.savefig("ess_plot.pdf", format="pdf")  

