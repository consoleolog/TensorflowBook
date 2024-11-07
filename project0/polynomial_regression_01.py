import os

import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_table(f'{os.getcwd()}/project0/data/income.txt')

df["income"].fillna( df["income"].mean(), inplace=True )

#opt 는 a,b,c 값 cov 는 공분산값임
opt, cov = curve_fit(lambda x, a, b, c: a * x + b * x**2 + c, df['age'], df['income'])

# a, b, c = opt

print(opt)