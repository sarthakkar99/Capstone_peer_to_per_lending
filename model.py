import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import timeit
import pickle as pickle
import warnings
from random import choice

warnings.filterwarnings("ignore")
data1= pd.read_csv("C:/Users/DELL/Desktop/Excel/credit1.csv")
data3=data1[['checking_balance','months_loan_duration','age','employment_length','amount','default']]
label_encoder=LabelEncoder()

columns=data3.columns
for cols in columns:
    if(isinstance(data3[cols].values[0],str)):
        data3[cols]=label_encoder.fit_transform(data3[cols].values)
x=data3.drop(['default'],axis=1).values
y=data3['default'].values
X_train,X_test,Y_train,Y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=False)
scaler=StandardScaler()
x_train=scaler.fit_transform(X_train)
x_test=scaler.transform(X_test)


svm=SVC()    
svm.fit(x_train,Y_train)    
y_pred=svm.predict(x_test)


lr=LogisticRegression()
lr.fit(x_train,Y_train)
y_pred=lr.predict(x_test)
 
kn=KNeighborsClassifier()
kn.fit(x_train,Y_train)
y_pred=kn.predict(x_test)


pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

pickle.dump(svm,open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))

pickle.dump(kn,open('model2.pkl','wb'))
model=pickle.load(open('model2.pkl','rb'))




