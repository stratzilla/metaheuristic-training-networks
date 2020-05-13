#!/usr/bin/env python3

import os
import glob
import pandas as pd
from math import floor
from sys import argv, exit

# go into directory
os.chdir(f'{argv[1]}')
# get filenames for all CSV in directory
filenames = [i for i in glob.glob('*.csv')]

# concatenate each CSV and store into a df
comb = pd.concat([pd.read_csv(f, header=None) for f in filenames], axis=1)
# name each column
comb.columns = [f'Run {i+1}' for i in range(len(comb.columns))]

# get the 40% worst performing runs
worst_columns = (comb.iloc[-1].nlargest(floor(len(comb.columns)*0.4)).index)
# and remove them from the df
comb = comb.drop(worst_columns, axis=1)

# calculate the mean of each row
comb['Mean'] = comb.mean(numeric_only=True, axis=1)
# add an epoch column
comb.insert(loc=0, column='Epoch', value=[i+1 for i in range(len(comb))])

# save as concatenated CSV file
comb.to_csv(f'{argv[2]}', index=False)
exit(0)