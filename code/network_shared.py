import pandas as pd
import matplotlib.pyplot as plt

def helper(epoch, mse, trp, tep, auto):
	"""Helper function.
	Outputs to console performance.

	Parameters:
		epoch : epoch to output.
		mse : mean squared error to output.
		trp : training accuracy to output.
		tep : testing accuracy to output.
		auto : flag for automated results collection.
	"""
	if not auto:
		print(f'{epoch}, {mse[-1]:.4f}, {trp[-1]:.2f}, {tep[-1]:.2f}')
	else:
		print(f'{mse[-1]:.4f}')

def load_data(filename):
	"""Loads CSV for splitting into training and testing data.

	Parameters:
		filename : the filename of the file to load.

	Returns:
		Two lists, each corresponding to training and testing data.
	"""
	# load into pandas dataframe
	df = pd.read_csv(filename, header=None, dtype=float)
	# normalize the data
	for features in range(len(df.columns)-1):
		df[features] = (df[features] - df[features].mean())/df[features].std()
	train = df.sample(frac=0.70).fillna(0.00) # get training portion
	test = df.drop(train.index).fillna(0.00) # remainder testing portion
	return train.values.tolist(), test.values.tolist()

def plot_data(epoch, mse, trp, tep):
	"""Plots data.
	Displays MSE, training accuracy, and testing accuracy over time.

	Parameters:
		epoch : how many epochs to plot over.
		mse : mean squared error over epochs.
		trp : training accuracy over epochs.
		tep : testing accuracy over epochs.
	"""
	x = range(0, epoch)
	fig, ax2 = plt.subplots()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('MSE', color='blue')
	line, = ax2.plot(x, mse, '-', c='blue', lw='1', label='MSE')
	ax1 = ax2.twinx()
	ax1.set_ylabel('Accuracy (%)', color='green')
	line2, = ax1.plot(x, trp, '-', c='green', lw='1', label='Training')
	line3, = ax1.plot(x, tep, ':', c='green', lw='1', label='Testing')
	fig.legend(loc='center')
	ax1.set_ylim(0, 101)
	plt.show()
	plt.clf()