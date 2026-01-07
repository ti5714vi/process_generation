
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
        this_time.append(float(line.split()[5]))
        this_error.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])
        this_rwgt.append(float(line.split()[31]))
        this_tot.append(float(line.split()[5])+float(line.split()[31])/float(line.split()[26]))

# Add last block if file does not end with empty line
if this_time:
    timings.append(np.array(this_time, dtype=float))
if this_error:
    errors[idx]=np.array(this_error, dtype=float)
if this_rwgt:
    rwgts.append(np.array(this_rwgt, dtype=float))
if this_time:
    totals.append(np.array(this_tot, dtype=float))

# fit-related things
def lin_model(x, C, b):
    return C + b*x
def exp_model(x, C, b):
    return math.exp(C) * (math.exp(b)**x)
fit_all=[]

log_timings=copy.copy(timings)
log_x = [math.log(r) for r in x]
for i in range(0,len(timings)):
    log_timings[i]=[math.log(r) for r in timings[i]]
    params, _ = curve_fit(lin_model, x, log_timings[i], p0=(1, 1.1))
    C_fit, b_fit = params
    fit_all.append([C_fit,b_fit])
x_fit = np.linspace(2,6, 200)

# latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# for shared axes
fig, axs = plt.subplots(
    2, 2,
    sharex=True,
    sharey=True,
    figsize=(9, 6)
)


# plot
num_subplots = 4  # 2x2
s = np.empty(4)
marksize=4
for idx in range(num_subplots):
    i = idx // 2  # row
    j = idx % 2   # column
    ax = axs[i, j]
    ax.set_xticks([2, 3, 4, 5, 6])
    if (idx == 0): tag=0
    if (idx == 1): tag=2
    if (idx == 2): tag=3
    if (idx == 3): tag=1
    s[idx] = format(math.exp(fit_all[tag][1]), ".1f")
    label="Fit $t \\sim$"+"$b^n$"
    ax.errorbar(x, timings[tag], yerr=errors[tag].T, marker='o', color=colors[idx],linestyle='', markersize=marksize,capsize=3, label="Total time")
    ax.plot(x,      rwgts[tag], marker='^', color=shade1[idx], linestyle='',markersize=marksize, label="Reweight time")
    ax.set_yscale('log')
    ax.tick_params(direction="in")
    ax.plot(x_fit, exp_model(x_fit, fit_all[tag][0], fit_all[tag][1]),linestyle='-',color=colors[idx], label=label, linewidth=0.7)

handle = ax.errorbar(
    [], [], yerr=[0.3],
    fmt='o',
    color='black',
    capsize=10,
    linestyle='none',
    markersize=marksize
)

handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 0, 1]
legend_handles = [
    Line2D([0], [0], marker='^', color='black',linestyle='',markersize=marksize,
           markerfacecolor='black', markeredgecolor='black', label='Reweight time'),
    Line2D([0], [0],  color='black',
           markerfacecolor='black', markeredgecolor='black', label='$t \\sim b^n$',linewidth=0.7),
    handle
]
fig.legend([legend_handles[i] for i in order],
           [labels[i] for i in order],
           loc="center right",
           bbox_to_anchor=(0.95, 0.5),
           frameon=True,
           ncol=1)

# labels on each plot
axs[0,0].text(0.1, 0.83, "$p p \\rightarrow nj$ \n $b=$"+str(s[0]), transform=axs[0,0].transAxes)
axs[0,1].text(0.1, 0.83, "$p p \\rightarrow t \\bar{t} + (n-2)j$ \n $b=$"+str(s[2]), transform=axs[0,1].transAxes)
axs[1,0].text(0.1, 0.83, "$p p \\rightarrow ZZ + (n-2)j$ \n $b=$"+str(s[3]), transform=axs[1,0].transAxes)
axs[1,1].text(0.1, 0.83, "$p p \\rightarrow e^+e^- + (n-2)j$ \n $b=$"+str(s[1]), transform=axs[1,1].transAxes)

upper_left = axs[0, 0].get_position()
lower_left = axs[1, 0].get_position()

axs[1, 0].set_xlabel(r"$n$")
axs[1, 1].set_xlabel(r"$n$")
axs[0, 0].set_ylabel(r"time $t$ [s]")
axs[1, 0].set_ylabel(r"time $t$ [s]")

# Improve layout
fig.tight_layout()
# Remove spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)
fig.subplots_adjust(right=0.68)

# Optional: show ticks only on outer axes
for ax in axs.flat:
    ax.label_outer()

# Save the plot as a PDF  
plt.savefig("timing_plot.pdf", format="pdf")  

