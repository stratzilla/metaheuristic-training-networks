#!/bin/bash

mkdir -p ../results/iris/bp
mkdir -p ../results/iris/ga
mkdir -p ../results/iris/pso
mkdir -p ../results/plots

printf "\nGetting results for BP-NN with Iris data set...";

for i in {1..50}; do
	../code/backprop_network.py iris 1 > ../results/iris/bp/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for GA-NN with Iris data set...";

for i in {1..50}; do
	../code/genetic_network.py iris 1 > ../results/iris/ga/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with Iris data set...";

for i in {1..50}; do
	../code/particle_network.py iris 1 > ../results/iris/pso/${i}.csv &
	if [ $(( $i % 10 )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

./concat_csv.py ../results/iris/bp/ ../../iris-bp.csv
./concat_csv.py ../results/iris/ga/ ../../iris-ga.csv
./concat_csv.py ../results/iris/pso/ ../../iris-pso.csv

sleep 1

printf " done! \nCreating a master list of runs...";

printf "BP-NN\n" > ../results/iris.csv
cat ../results/iris-bp.csv >> ../results/iris.csv
printf "\nGA-NN\n" >> ../results/iris.csv
cat ../results/iris-ga.csv >> ../results/iris.csv
printf "\nPSO-NN\n" >> ../results/iris.csv
cat ../results/iris-pso.csv >> ../results/iris.csv

sleep 1

printf " done! \nMaking plots...";

./plot_iris.gp

printf " done! \nCleaning up...";

rm -r ../results/iris
rm -r ../results/iris-bp.csv
rm -r ../results/iris-ga.csv
rm -r ../results/iris-pso.csv

printf " done! \nCompleted collecting Iris data!\n\n";