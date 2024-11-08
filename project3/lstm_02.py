import os

import numpy as np
import pandas as pd

# from urllib.request import urlretrieve

# urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', f'{os.getcwd()}/project3/data/shopping.txt')

df = pd.read_table(f"{os.getcwd()}/project3/data/shopping.txt", names=['rating', 'review'])

df['label'] = np.where(df['rating'] > 3, 1, 0)

df['review'] = df['review'].str.replace("[ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]","")

