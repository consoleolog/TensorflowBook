import os

import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_table(f'{os.getcwd()}/project0/data/income.txt')

x = np.column_stack([ df['age'], df['age']**2, np.ones(len(df)) ])
model = sm.OLS(df['income'], x).fit()
print(model.summary())