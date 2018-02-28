'''
    About: This script is used simply to test a few libraries that are useful to do some Data Science
    
    Attention: This script must be executed from within iPython:
        1 - ipython
        2 - run script_name.py
'''

# 1 - Test the imports
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import sklearn
import requests
import bs4
import seaborn as sns

x = np.arange(1,50)
y = np.array(list(map(lambda a: np.sin(a) + np.log2(a), x)))

print(x)
print(y)

# Ignore the error, the get_ipython() method is going to be available if the script is executed 
# from within iPython
get_ipython().run_line_magic('matplotlib', 'osx')

# 2 - Plot a simple chart
import matplotlib.pyplot as plt
plt.plot(x, y)

# 3 - Plot a slightly more elaborated chart 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
from numpy.random import normal

# Set the mean and standard deviation
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 30, normed=True)

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)),
         linewidth=2, color='r')
plt.show()

# 4 - Data datasets concatenation
A = pd.Series(["A{}".format(a) for a in range(4)], index=range(4), name="A")
B = pd.Series(["B{}".format(a) for a in range(4)], index=range(4), name= "B")
C = pd.Series(["C{}".format(a) for a in range(5)], index=range(5), name= "C")
pd.concat([A,B,C],axis=1)