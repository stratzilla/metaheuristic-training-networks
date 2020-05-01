#!/bin/bash

# confirm script is run with proper data set as argument
case $1 in
	"iris")
		data_name="Iris";;
	"wheat")
		data_name="Wheat Seeds";;
	"wine")
		data_name="Wine";;
	"breast")
		data_name="Breast Cancer";;
	"all") # if running all results collections at once
		printf "\nCollecting results for all data sets...\n";
		sleep 0.5
		./results_collection.sh iris
		sleep 0.5
		./results_collection.sh wheat
		sleep 0.5
		./results_collection.sh wine
		sleep 0.5
		./results_collection.sh breast
		sleep 0.5
		printf "Completed results collection for all data sets!\n\n";
		exit 0;;
	*) # if data doesn't exist or no argument
		if [ "$1" == "" ]; then
			error="No data set was selected."
		else
			error="Data set \"$1\" not supported."
		fi
		printf "\n$error\n";
		printf "\nExecute this script as:\n"; 
		printf " $ ./results_collection.sh <data>\n";
		printf "\nWhere available data sets (case-sensitive) are:\n";
		printf " - \"iris\" for Iris data set\n";
		printf " - \"wheat\" for Wheat Seeds data set\n";
		printf " - \"wine\" for Wine data set\n";
		printf " - \"breast\" for Breast Cancer data set\n";
		printf " - \"all\" for all of the data sets above\n\n";
		exit 1;;
esac

# remove old results
rm -rf ../results/${1}/bp
rm -rf ../results/${1}/ga
rm -rf ../results/${1}/pso

sleep 0.5

# make directories to hold data
mkdir -p ../results/${1}/bp
mkdir -p ../results/${1}/ga
mkdir -p ../results/${1}/pso
mkdir -p ../results/plots

max_runs=50
max_concurrent_runs=10

printf "\nGetting results for BP-NN with $data_name data set...";

# collect BP-NN data
for i in $(seq 1 $max_runs); do
	../code/backprop_network.py ${1} 1 > ../results/${1}/bp/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		# only train ten networks at once
		wait
	fi
done

printf " done! \nGetting results for GA-NN with $data_name data set...";

# collect GA-NN data
for i in $(seq 1 $max_runs); do
	../code/genetic_network.py ${1} 1 > ../results/${1}/ga/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with $data_name data set...";

# collect PSO-NN data
for i in $(seq 1 $max_runs); do
	../code/particle_network.py ${1} 1 > ../results/${1}/pso/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

# concatenate all runs into one CSV file
./concat_csv.py ../results/${1}/bp/ ../../${1}-bp.csv
./concat_csv.py ../results/${1}/ga/ ../../${1}-ga.csv
./concat_csv.py ../results/${1}/pso/ ../../${1}-pso.csv

sleep 1

printf " done! \nCreating a master list of runs...";

# concatenate all CSV files into one master file
printf "BP-NN\n" > ../results/${1}.csv
cat ../results/${1}-bp.csv >> ../results/${1}.csv
printf "\nGA-NN\n" >> ../results/${1}.csv
cat ../results/${1}-ga.csv >> ../results/${1}.csv
printf "\nPSO-NN\n" >> ../results/${1}.csv
cat ../results/${1}-pso.csv >> ../results/${1}.csv

sleep 1

printf " done! \nMaking plots...";

# create plot of MSE over epochs
gnuplot -e "res='$1'" plot_results.gp

sleep 0.5

printf " done! \nCleaning up...";

# clean up redundant files
rm -r ../results/${1}
rm -r ../results/${1}-bp.csv
rm -r ../results/${1}-ga.csv
rm -r ../results/${1}-pso.csv

sleep 0.5

printf " done! \nCompleted collecting $data_name data!\n\n";

exit 0