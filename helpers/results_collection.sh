#!/bin/bash

# confirm script is run with proper data set as argument
case $1 in
	"iris")
		data_name="Iris";;
	"penguins")
		data_name="Penguins";;
	"wheat")
		data_name="Wheat Seeds";;
	"wine")
		data_name="Wine";;
	"breast")
		data_name="Breast Cancer";;
	"ionosphere")
		data_name="Ionosphere Radar";;
	"all") # if running all results collections at once
		printf "\nCollecting results for all data sets...\n";
		sleep 0.5
		./results_collection.sh iris
		sleep 0.5
		./results_collection.sh penguins
		sleep 0.5
		./results_collection.sh wheat
		sleep 0.5
		./results_collection.sh wine
		sleep 0.5
		./results_collection.sh breast
		sleep 0.5
		./results_collection.sh ionosphere
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
		printf " - \"penguins\" for Penguins data set\n";
		printf " - \"wheat\" for Wheat Seeds data set\n";
		printf " - \"wine\" for Wine data set\n";
		printf " - \"breast\" for Breast Cancer data set\n";
		printf " - \"ionosphere\" for Ionosphere Radar data set\n";
		printf " - \"all\" for all of the data sets above\n\n";
		exit 1;;
esac

# remove old results
rm -rf ../results/temp/
rm -rf ../results/csv/${1}.csv
rm -rf ../results/plots/${1}-plot.png
rm -rf ../results/statistics/${1}.txt

sleep 0.5

# make directories to hold data
mkdir -p ../results/temp/bp
mkdir -p ../results/temp/ga
mkdir -p ../results/temp/pso
mkdir -p ../results/temp/de
mkdir -p ../results/temp/ba
mkdir -p ../results/csv
mkdir -p ../results/plots
mkdir -p ../results/statistics

sleep 0.5

max_runs=100
max_concurrent_runs=10

printf "\nGetting results for BP-NN with $data_name data set...";

# collect BP-NN data
for i in $(seq 1 $max_runs); do
	../code/backprop_network.py ${1} 1 > ../results/temp/bp/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		# only train ten networks at once
		wait
	fi
done

printf " done! \nGetting results for GA-NN with $data_name data set...";

# collect GA-NN data
for i in $(seq 1 $max_runs); do
	../code/genetic_network.py ${1} 1 > ../results/temp/ga/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for PSO-NN with $data_name data set...";

# collect PSO-NN data
for i in $(seq 1 $max_runs); do
	../code/particle_network.py ${1} 1 > ../results/temp/pso/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for DE-NN with $data_name data set...";

# collect DE-NN data
for i in $(seq 1 $max_runs); do
	../code/evolve_network.py ${1} 1 > ../results/temp/de/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

printf " done! \nGetting results for BA-NN with $data_name data set...";

# collect BA-NN data
for i in $(seq 1 $max_runs); do
	../code/bat_network.py ${1} 1 > ../results/temp/ba/${i}.csv &
	if [ $(( $i % $max_concurrent_runs )) == 0 ]; then
		wait
	fi
done

wait

printf " done! \nConcatenating results...";

# concatenate all runs into one CSV file
./concat_csv.py ../results/temp/bp/ ../../temp/${1}-bp.csv
./concat_csv.py ../results/temp/ga/ ../../temp/${1}-ga.csv
./concat_csv.py ../results/temp/pso/ ../../temp/${1}-pso.csv
./concat_csv.py ../results/temp/de/ ../../temp/${1}-de.csv
./concat_csv.py ../results/temp/ba/ ../../temp/${1}-ba.csv

sleep 1

printf " done! \nCreating a master list of runs...";

# concatenate all CSV files into one master file
printf "BP-NN\n" > ../results/csv/${1}.csv
cat ../results/temp/${1}-bp.csv >> ../results/csv/${1}.csv
printf "\nGA-NN\n" >> ../results/csv/${1}.csv
cat ../results/temp/${1}-ga.csv >> ../results/csv/${1}.csv
printf "\nPSO-NN\n" >> ../results/csv/${1}.csv
cat ../results/temp/${1}-pso.csv >> ../results/csv/${1}.csv
printf "\nDE-NN\n" >> ../results/csv/${1}.csv
cat ../results/temp/${1}-de.csv >> ../results/csv/${1}.csv
printf "\nBA-NN\n" >> ../results/csv/${1}.csv
cat ../results/temp/${1}-ba.csv >> ../results/csv/${1}.csv

sleep 1

printf " done! \nPerforming statistical tests...";

# run R script to perform Anova, Tukey HSD
Rscript statistics.r ${1} > ../results/statistics/${1}.txt

sleep 0.5

printf " done! \nMaking plots...";

# create plot of MSE over epochs
gnuplot -e "res='$1'" plot_results.gp

sleep 0.5

printf " done! \nCleaning up...";

# clean up redundant files
rm -rf ../results/temp

sleep 1

printf " done! \nCompleted collecting $data_name data!\n\n";

exit 0
