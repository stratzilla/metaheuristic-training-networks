#!/bin/bash

mkdir -p ../results/breast/bp
mkdir -p ../results/breast/ga
mkdir -p ../results/breast/pso
mkdir -p ../results/plots

printf "\nGetting results for BP-NN with Breast Cancer data set...";

for i in {1..50}; do
	../code/backprop_network.py breast 1 > ../results/breast/bp/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for GA-NN with Breast Cancer data set...";

for i in {1..50}; do
	../code/genetic_network.py breast 1 > ../results/breast/ga/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with Breast Cancer data set...";

for i in {1..50}; do
	../code/particle_network.py breast 1 > ../results/breast/pso/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

./concat_csv.py ../results/breast/bp/ ../../breast-bp.csv
./concat_csv.py ../results/breast/ga/ ../../breast-ga.csv
./concat_csv.py ../results/breast/pso/ ../../breast-pso.csv

sleep 1

printf " done! \nCreating a master list of runs...";

printf "BP-NN\n" > ../results/breast.csv
cat ../results/breast-bp.csv >> ../results/breast.csv
printf "\nGA-NN\n" >> ../results/breast.csv
cat ../results/breast-ga.csv >> ../results/breast.csv
printf "\nPSO-NN\n" >> ../results/breast.csv
cat ../results/breast-pso.csv >> ../results/breast.csv

sleep 1

printf " done! \nMaking plots...";

./plot_breast.gp

printf " done! \nCleaning up...";

rm -r ../results/breast
rm -r ../results/breast-bp.csv
rm -r ../results/breast-ga.csv
rm -r ../results/breast-pso.csv

printf " done! \nCompleted collecting Breast Cancer data!\n\n";