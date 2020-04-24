#!/bin/bash

mkdir -p ../results/wine/bp
mkdir -p ../results/wine/ga
mkdir -p ../results/wine/pso
mkdir -p ../results/plots

printf "\nGetting results for BP-NN with Wine data set...";

for i in {1..50}; do
	../code/backprop_network.py wine 1 > ../results/wine/bp/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for GA-NN with Wine data set...";

for i in {1..50}; do
	../code/genetic_network.py wine 1 > ../results/wine/ga/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with Wine data set...";

for i in {1..50}; do
	../code/particle_network.py wine 1 > ../results/wine/pso/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

./concat_csv.py ../results/wine/bp/ ../../wine-bp.csv
./concat_csv.py ../results/wine/ga/ ../../wine-ga.csv
./concat_csv.py ../results/wine/pso/ ../../wine-pso.csv

sleep 1

printf " done! \nCreating a master list of runs...";

printf "BP-NN\n" > ../results/wine.csv
cat ../results/wine-bp.csv >> ../results/wine.csv
printf "\nGA-NN\n" >> ../results/wine.csv
cat ../results/wine-ga.csv >> ../results/wine.csv
printf "\nPSO-NN\n" >> ../results/wine.csv
cat ../results/wine-pso.csv >> ../results/wine.csv

sleep 1

printf " done! \nMaking plots...";

./plot_wine.gp

printf " done! \nCleaning up...";

rm -r ../results/wine
rm -r ../results/wine-bp.csv
rm -r ../results/wine-ga.csv
rm -r ../results/wine-pso.csv

printf " done! \nCompleted collecting Wine data!\n\n";