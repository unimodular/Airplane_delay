# Airplane_delay

### Yunze Wang, Shuo Li, Xinrui Zhong

This repository is used for STAT 628 module_3 Airplane_delay Study Assignment. All of our works are included in this repository.


## Overview
This project aims to analyze patterns in flight delays and cancellations by integrating data from the U.S. Department of Transportation and the U.S. National Weather Service. The dataset includes historical records of flight delays, cancellations, and weather conditions. This analysis is focused on the holiday season (November to January) to identify trends that could help passengers avoid delays and cancellations. Additionally, the project involves building a predictive model to provide insights into potential delays and cancellation risks.

## Requirements
To run the codes, you will need the following libraries installed in your environment. The code is in the 

```python


# Install necessary libraries

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from haversine import haversine, Unit
from timezonefinder import TimezoneFinder


```



You can install these dependencies using `pip`:

```bash
pip install numpy pandas statsmodels matplotlib seaborn scikit-learn
```
