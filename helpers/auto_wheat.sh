#!/bin/bash

mkdir -p ../results/wheat/bp
mkdir -p ../results/wheat/ga
mkdir -p ../results/wheat/pso
mkdir -p ../results/plots

printf "\nGetting results for BP-NN with Wheat Seeds data set...";

for i in {1..50}; do
	../code/backprop_network.py wheat 1 > ../results/wheat/bp/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for GA-NN with Wheat Seeds data set...";

for i in {1..50}; do
	../code/genetic_network.py wheat 1 > ../results/wheat/ga/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with Wheat Seeds data set...";

for i in {1..50}; do
	../code/particle_network.py wheat 1 > ../results/wheat/pso/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

./concat_csv.py ../results/wheat/bp/ ../../wheat-bp.csv
./concat_csv.py ../results/wheat/ga/ ../../wheat-ga.csv
./concat_csv.py ../results/wheat/pso/ ../../wheat-pso.csv

sleep 1

printf " done! \nCreating a master list of runs...";

printf "BP-NN\n" > ../results/wheat.csv
cat ../results/wheat-bp.csv >> ../results/wheat.csv
printf "\nGA-NN\n" >> ../results/wheat.csv
cat ../results/wheat-ga.csv >> ../results/wheat.csv
printf "\nPSO-NN\n" >> ../results/wheat.csv
cat ../results/wheat-pso.csv >> ../results/wheat.csv

sleep 1

printf " done! \nMaking plots...";

./plot_wheat.gp

printf " done! \nCleaning up...";

rm -r ../results/wheat
rm -r ../results/wheat-bp.csv
rm -r ../results/wheat-ga.csv
rm -r ../results/wheat-pso.csv

printf " done! \nCompleted collecting Wheat Seeds data!\n\n";