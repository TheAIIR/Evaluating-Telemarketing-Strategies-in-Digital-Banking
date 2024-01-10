# Evaluating-Telemarketing-Strategies-in-Digital-Banking

# Introduction
The goal of our M4RAI project is to forecast customer behaviors by analyzing previous outcomes of marketing campaigns, based on socio-economic variables with the use of machine learning techniques.

# Requirements
```
from sklearn.preprocessing import StandardScaler
from scipy import stats as sts
import pandas as pd
from scipy.stats.distributions import randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
import lime as lime
from lime.lime_tabular import LimeTabularExplainer
import shap as shap
import pickle
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
```
