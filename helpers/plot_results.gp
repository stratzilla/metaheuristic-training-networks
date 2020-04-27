#!/usr/bin/gnuplot

# verify argument was passed
if (!exists("res")) { exit 1 }

# create PNG output
set terminal png
set output "../results/plots/".res."-plot.png"

# set parameters for MSE lines later
set parametric
set trange [0.005:0.030]

# set axis titles
set xlabel "Epoch"
set ylabel "MSE" offset 2

# separate CSV by commas
set datafile separator comma

# load CSV as lines
bp = "../results/".res."-bp.csv" 
ga = "../results/".res."-ga.csv"
pso = "../results/".res."-pso.csv"

# get column count
cols_bp = int(system('head -1 '.bp.' | wc -w'))+1
cols_ga = int(system('head -1 '.ga.' | wc -w'))+1
cols_pso = int(system('head -1 '.pso.' | wc -w'))+1

# position where MSE <= 0.1
delimiter = '-F "\"*,\"*"'
call = "awk ".delimiter." 'NR>1 && $NF<=0.1 {n=$1; exit} END{print n+0}' "
mse_bp = int(system(call.bp))
mse_ga = int(system(call.ga))
mse_pso = int(system(call.pso))

# do not plot if MSE <= 0.1 was never reached
mse_bp = (mse_bp == 0 ? NaN : mse_bp)
mse_ga = (mse_ga == 0 ? NaN : mse_ga)
mse_pso = (mse_pso == 0 ? NaN : mse_pso)

# plot each line
plot bp using cols_bp with lines linecolor 1 linewidth 3 title "BP-NN", \
	 ga using cols_ga with lines linecolor 2 linewidth 3 title "GA-NN", \
	 pso using cols_pso with lines linecolor 3 linewidth 3 title "PSO-NN", \
	 mse_bp,t linecolor 1 linewidth 6 title "", \
	 mse_ga,t linecolor 2 linewidth 6 title "", \
	 mse_pso,t linecolor 3 linewidth 6 title ""