
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
    else:
        this_time.append(float(line.split()[11]))
        this_error.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])

idx=0
for line in lines2:
    if line.strip() == "":  # empty row signals end of block
        if this_time2:
            timings2.append(np.array(this_time2, dtype=float))
            this_time2 = []
            errors2[idx]=np.array(this_error2, dtype=float)
            idx=idx+1
            this_error2 = []
    else:
        this_time2.append(float(line.split()[11]))
        this_error2.append([float(line.split()[5])-float(line.split()[6]),\
                          float(line.split()[7])-float(line.split()[5])])

# Add last block if file does not end with empty line
if this_time:
    timings.append(np.array(this_time, dtype=float))
if this_error:
    errors[idx]=np.array(this_error, dtype=float)

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
    1, 2,
    sharex=True,
    sharey=True,
    figsize=(6, 3),
    constrained_layout=False
)


# plot
num_subplots = 2  # 2x2
s = np.empty(2)
for idx in range(num_subplots):
    ax = axs[idx]
    if (idx == 0): tag=0
    if (idx == 1): tag=1
    s[idx] = format(math.exp(fit_all[tag][1]), ".1f")
    ax.tick_params(direction="in")
    ax.plot(x, timings[tag]/timings2[tag], marker='o', color=shade1[idx],linestyle='', markersize=2,  label="Ratioe")

fig = plt.gcf()
handles, labels = plt.gca().get_legend_handles_labels()

# labels on each plot
axs[0].text(0.1, 0.85, "$p p \\rightarrow nj$", transform=axs[0].transAxes)
axs[1].text(0.1, 0.85, "$p p \\rightarrow t \\bar{t} + (n-2)j$", transform=axs[1].transAxes)

upper_left = axs[0].get_position()
lower_left = axs[1].get_position()

axs[0].set_xlabel(r"$n$")
axs[1].set_xlabel(r"$n$")
axs[0].set_ylabel(r"$\frac{\sigma_{\mathrm{3qq}}}{\sigma_{\mathrm{no-3qq}}}$")

# Improve layout
fig.tight_layout()
# Remove spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Optional: show ticks only on outer axes
for ax in axs.flat:
    ax.label_outer()

# Save the plot as a PDF  
plt.savefig("3qq_xsec_plot.pdf", format="pdf")  

