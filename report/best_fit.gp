#!/usr/bin/gnuplot

#
# this program was used for report
#

# create PNG output
set terminal png
set output "../report/best-fit.png"

# set ranges
set xrange [0:400]
set yrange [0:100]

# turn legend off
set key off

# set axis titles
set xlabel "Dimensionality"
set ylabel "Avg # of Epochs" offset 2

# styles for each training method
set style line 1 linecolor '#9400d3' linewidth 2 pt 7
set style line 2 linecolor '#009e74' linewidth 2 pt 7
set style line 3 linecolor '#56b3e9' linewidth 2 pt 7
set style line 4 linecolor '#e69d00' linewidth 2 pt 7
set style line 5 linecolor '#e51e10' linewidth 2 pt 7

# lines of best fit
b(x) = -0.1*x + 47.51
g(x) = 0.24*x + 17.46
p(x) = -0.05*x + 32.41
d(x) = -0.05*x + 71.78
a(x) = 0*x + 51.24

# plot each line
plot "<echo '27, 70'" with points ls 1, \
	"<echo '43, 29'" with points ls 1, \
	"<echo '58, 49'" with points ls 1, \
	"<echo '105, 17'" with points ls 1, \
	"<echo '274, 9'" with points ls 1, \
	"<echo '372, 22'" with points ls 1, \
	"<echo '27, 31'" with points ls 2, \
	"<echo '43, 14'" with points ls 2, \
	"<echo '58, 38'" with points ls 2, \
	"<echo '105, 43'" with points ls 2, \
	"<echo '27, 27'" with points ls 3, \
	"<echo '43, 19'" with points ls 3, \
	"<echo '58, 52'" with points ls 3, \
	"<echo '105, 21'" with points ls 3, \
	"<echo '274, 20'" with points ls 3, \
	"<echo '27, 82'" with points ls 4, \
	"<echo '43, 49'" with points ls 4, \
	"<echo '58, 85'" with points ls 4, \
	"<echo '105, 58'" with points ls 4, \
	"<echo '274, 61'" with points ls 4, \
	"<echo '27, 67'" with points ls 5, \
	"<echo '43, 37'" with points ls 5, \
	"<echo '58, 60'" with points ls 5, \
	"<echo '105, 45'" with points ls 5, \
	"<echo '274, 32'" with points ls 5, \
	"<echo '372, 66'" with points ls 5, \
	b(x) with lines ls 1, \
	g(x) with lines ls 2, \
	p(x) with lines ls 3, \
	d(x) with lines ls 4, \
	a(x) with lines ls 5