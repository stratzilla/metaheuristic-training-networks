#!/usr/bin/env python3

from math import floor
from sys import argv, exit
from os import chdir
from glob import glob
import pandas as pd

# go into directory
chdir(f'{argv[1]}')
# get filenames for all CSV in directory
FILENAMES = [i for i in glob('*.csv')]

# concatenate each CSV and store into a df
COMB = pd.concat([pd.read_csv(f, header=None) for f in FILENAMES], axis=1)
# name each column
COMB.columns = [f'Run {i+1}' for i in range(len(COMB.columns))]

# get the 20% worst performing runs
WORST_COLUMNS = (COMB.iloc[-1].nlargest(floor(len(COMB.columns)*0.2)).index)
# and remove them from the df
COMB = COMB.drop(WORST_COLUMNS, axis=1)

# calculate the mean of each row
COMB['Mean'] = COMB.mean(numeric_only=True, axis=1)
# add an epoch column
COMB.insert(loc=0, column='Epoch', value=[i+1 for i in range(len(COMB))])

# round every value to six decimal place precision
COMB = COMB.round(6)

# save as concatenated CSV file
COMB.to_csv(f'{argv[2]}', index=False)
exit(0)
