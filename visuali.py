import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

file = pd.read_csv('tBLG_Data.csv')
df=file
#查看数据类别个数
df['_Target'].value_counts()
df['Species_num'] = df['_Target'].map({'SLG':1,'0-9°':2,'9-20°':3,'20-30°':4})
df = df.drop(columns = ['_Target'])
print(df)