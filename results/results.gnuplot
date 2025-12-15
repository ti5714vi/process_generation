#################################################################################
# command:
# gnuplot results.gnuplot && pdflatex plot && open plot.pdf
#################################################################################



reset

# Make the canvas large enough for 4 panels
# (adjust if you need a specific size in your document)
set terminal epslatex color size 16cm,11cm

# --- Styles (unchanged) ---
set key bottom Right samplen 1.4

set style line  9  lc rgb 'gray'   lw 3.0 
set style line 99 lc rgb 'black'   lw 3.0 
set style line  1 lc rgb '#0d6986' lw 5.0
set style line  2 lc rgb '#660d86' lw 5.0
set style line  3 lc rgb '#862a0d' lw 5.0
set style line  4 lc rgb '#2d860d' lw 5.0
set style line 11 lc rgb '#0d6986' lw 5.0 dt 2.1
set style line 12 lc rgb '#660d86' lw 5.0 dt 2.1
set style line 13 lc rgb '#862a0d' lw 5.0 dt 2.1
set style line 14 lc rgb '#2d860d' lw 5.0 dt 2.1

set xtics scale 1.5
set ytics scale 1.5

################################################################################
# Shared axes setup
set xrange [1.8:6.2]
set mytics 10
set xtics 1

set yrange [5:30000]
set logscale y

# We'll only show y-labels on the left column and x-labels on the bottom row.
XLabelStr = "$n$"

# Optional: tighten panel margins a bit
set bmargin 0.1
set tmargin 0.1
set lmargin 0.1
set rmargin 0.1

################################################################################
# Fits (unchanged)
f1(x) = a1*x + b1
fit f1(x) 'results.dat' index 1 using 2:(log($6)):( (log($8)-log($7))/2 ) via a1,b1

f2(x) = a2*x + b2
fit f2(x) 'results.dat' index 2 using ($2+1):(log($6)):( (log($8)-log($7))/2 ) via a2,b2

f3(x) = a3*x + b3
fit f3(x) 'results.dat' index 3 using ($2+2):(log($6)):( (log($8)-log($7))/2 ) via a3,b3

f4(x) = a4*x + b4
fit f4(x) 'results.dat' index 4 using ($2+2):(log($6)):( (log($8)-log($7))/2 ) via a4,b4

# Convert back to exponential parameters
A1 = exp(b1); B1 = a1
A2 = exp(b2); B2 = a2
A3 = exp(b3); B3 = a3
A4 = exp(b4); B4 = a4

# Plot functions
g1(x) = A1*exp(B1*x)
g2(x) = A2*exp(B2*x)
g3(x) = A3*exp(B3*x)
g4(x) = A4*exp(B4*x)

###############################################################################
# TOTAL TIME FOR LC (with fit)
################################################################################
# Multiplot: 2 rows x 2 columns
set output "LCtime.tex"
YLabelStr = "time $t$ [s]"
set multiplot layout 2,2 title "Time to generate 100k unweighted LC events" font ",12" offset 0,1

# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "$10^{%T}$"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):6:7:8 with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets; $t\\approx c \\cdot %.1f^{n}$", exp(B1)), \
  g1(x) with lines ls 1 lw 2 notitle

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):6:7:8 with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets; $t\\approx c \\cdot %.1f^{n}$", exp(B2)), \
  g2(x) with lines ls 2 lw 2 notitle

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "$10^{%T}$"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):6:7:8 with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets; $t\\approx c \\cdot %.1f^{n}$", exp(B3)), \
  g3(x) with lines ls 3 lw 2 notitle

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):6:7:8 with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets; $t\\approx c \\cdot %.1f^{n}$", exp(B4)), \
  g4(x) with lines ls 4 lw 2 notitle

unset multiplot


################################################################################
# LC cross section
################################################################################
# Multiplot: 2 rows x 2 columns
set output "LCxsec.tex"
set multiplot layout 2,2 title "LC cross section" font ",12" offset 0,1

YLabelStr = "Cross section $\\sigma$ [pb]"
set yrange [0.01:1000000000]


# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "$10^{%T}$"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):12:13:14 with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets")

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):12:13:14 with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets")

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "$10^{%T}$"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):12:13:14 with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets")

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):12:13:14 with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets")

unset multiplot

################################################################################
# secondary unweighting efficiency
################################################################################
# Multiplot: 2 rows x 2 columns
set output "2ndUnwEff.tex"
set multiplot layout 2,2 title "LC $\\to$ FC unweighting efficiency" font ",12" offset 0,1

YLabelStr = "Efficiency $\\epsilon_2$"
set yrange [0.5:1.05]

# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
unset logscale y
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "%g"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):26:27:28 with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets")

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):26:27:28 with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets")

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "%g"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):26:27:28 with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets")

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):26:27:28 with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets")

unset multiplot

################################################################################
# Effectve sample size fraction (after reweighting)
################################################################################
# Multiplot: 2 rows x 2 columns
set output "ESS.tex"
set multiplot layout 2,2 title "Kish effective sample size fraction" font ",12" offset 0,1

YLabelStr = "$f_{\\mathrm{ESS}}$"
set yrange [0.978:1.002]

# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
unset logscale y
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "%g"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):($29/100000):($30/100000):($31/100000) with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets")

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):($29/100000):($30/100000):($31/100000) with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets")

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "%g"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):($29/100000):($30/100000):($31/100000) with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets")

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):($29/100000):($30/100000):($31/100000) with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets")

unset multiplot

################################################################################
# LC -> FC average reweight factor
################################################################################
# Multiplot: 2 rows x 2 columns
set output "LCFCrwgt.tex"
set multiplot layout 2,2 title "Average LC $\\to$ FC reweight factor" font ",12" offset 0,1

YLabelStr = "$\\langle f_{\\mathrm{LC}\\to\\mathrm{FC}} \\rangle$"
set yrange [0.6:1.002]

# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
unset logscale y
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "%g"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):38:39:40 with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets")

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):38:39:40 with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets")

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "%g"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):38:39:40 with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets")

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):38:39:40 with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets")

unset multiplot

################################################################################
# Time for reweighting
################################################################################
# Multiplot: 2 rows x 2 columns
set output "LCFCtime.tex"
set multiplot layout 2,2 title "Time to reweight LC $\\to$ FC" font ",12" offset 0,1

YLabelStr = "time [s]"
set yrange [2:2000]
set logscale y

# ----------------------------
# Panel (row 1, col 1): index 1
# Show y-label (left column), hide x-label (top row)
set ylabel YLabelStr offset +1,0
unset xlabel
set ytics format "$10^{%T}$"
set xtics format ""
plot \
  'results.dat' index 1 using ($2+0.05):32:33:34 with yerrorbars ls 1 pt 0 \
      title sprintf("$pp \\to n$ jets")

# ----------------------------
# Panel (row 1, col 2): index 2
# Hide y-labels/tics (right column), hide x-label (top row)
unset ylabel
set ytics format ""
set xtics format ""
unset xlabel
plot \
  'results.dat' index 2 using ($2+1):32:33:34 with yerrorbars ls 2 pt 0 \
      title sprintf("$pp \\to e^+e^-+(n-1)$ jets")

# ----------------------------
# Panel (row 2, col 1): index 3
# Show y-labels (left column), show x-label (bottom row)
set ylabel YLabelStr offset +1,0
set ytics format "$10^{%T}$"
set xtics format "%g"
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 3 using ($2+2):32:33:34 with yerrorbars ls 3 pt 0 \
      title sprintf("$pp \\to t\\bar{t}+(n-2)$ jets")

# ----------------------------
# Panel (row 2, col 2): index 4
# Hide y-labels/tics (right column), show x-label (bottom row)
unset ylabel
set ytics format ""
set xlabel XLabelStr offset 0,0.5
plot \
  'results.dat' index 4 using ($2+1.95):32:33:34 with yerrorbars ls 4 pt 0 \
      title sprintf("$pp \\to ZZ+(n-2)$ jets")

unset multiplot

################################################################################
