import pandas as pd
from os.path import exists
import numpy as np
from scipy.stats import *


def get_stats(x, y, degree=1):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    yhat = p(x)  
    ybar = np.sum(y)/len(y)         
    results['reg'] = np.sum((yhat-ybar)**2) 
    results['r2'] = np.corrcoef(x, y)[0, 1]**2
    results['pearson'] = pearsonr(x,y)

    return results
