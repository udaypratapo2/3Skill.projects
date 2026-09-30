import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np

from sklearn.metrics import roc_curve , auc , roc_auc_scor

data = pd.read_csv("dataset_12000_records.csv")

x=data.drop(column = "Readmitted_30_Days")
y=data["Readmitted_30_Days"]


#print(data.shape)
#print(data.describe())
data.info()