# Comparative Study of Metaheuristic Training Networks

This repository was used for a comparative study between backpropagation trained feed forward networks and the same using genetic algorithms and particle swarm optimization in lieu of backpropagation.

I was interested in using PSO to train a neural network and decided to compare it, along with GA, to BP. Results are interesting: PSO-NN works very well for smaller data but BP-NN is the clear winner in higher dimensional data. GA-NN was found to be impractical or inefficient for training irrespective of problem size.

# Dependencies

- Python 3.6
- GNU/Linux
- gnuplot
- X11 or similar
- `pandas`
- `matplotlib`

# Execution

You can clone this repository to your computer like so:

` $ git clone https://www.github.com/stratzilla/metaheuristic-training-networks`

You can run any of the training scripts in `/code` like so:

```bash
 $ ./backprop_network.py <arg>
 $ ./genetic_network.py <arg>
 $ ./particle_network.py <arg>
```

This will train a single network and output a `matplot` plot showing the mean squared error over epochs as well as training and testing accuracy per epoch. 

If you want to automate training multiple networks to find the mean between training runs, you can instead execute the below from `/helpers`:

```bash
 $ ./results_collection.sh <arg>
```

This will train fifty networks of each type, concatenate the results, and use `gnuplot` to make a master plot comparing each training technique. You can edit this file manually to change how many runs per network and how many concurrent runs (default is `50` and `10`).

These scripts use these for `<arg>`:

- `iris` for Iris data set
- `wheat` for Wheat Seeds data set
- `wine` for Wine data set
- `breast` for Breast Cancer data set

These data sets are defined in `/data`.

The algorithms are data agnostic and will take any data, you just need to preprocess data to be accepted: final columnar value for data is the classification while the others are attributes. Classes must be enumerated starting at `0`, and data should be numerical (continuous or integer).

# Results

You can see CSV results in `/results` or for a visualization of mean squared error over epochs for each network, you can see these in `/img`.
