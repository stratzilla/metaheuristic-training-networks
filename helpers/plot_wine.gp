#!/usr/bin/gnuplot -persist

set terminal png
set output "../results/plots/wine-plot.png"
set xlabel "Epoch"
set ylabel "MSE" offset 2
set datafile separator comma
bp = "../results/wine-bp.csv" 
ga = "../results/wine-ga.csv"
pso = "../results/wine-pso.csv"
plot bp using 31 with lines linecolor 1 linewidth 3 title "BP-NN" ,\
	 ga using 31 with lines linecolor 2 linewidth 3 title "GA-NN" ,\
	 pso using 31 with lines linecolor 3 linewidth 3 title "PSO-NN"