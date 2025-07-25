# from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DescitionTrees import DecisionTree
from RandomForest import RandomForest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# data=datasets.load_breast_cancer()
# X,y=data.data,data.target

# X_train, X_test, y_train,y_test = train_test_split(
#     X,y, test_size=0.2,random_state=1234
# )

# clf=DecisionTree()
# clf.fit(X_train,y_train)
# prediction=clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

# acc=accuracy(y_test, prediction)
# print(acc)

# clf=RandomForest()
# clf.fit(X_train,y_train)
# predictions=clf.predict(X_test)

# acc= accuracy(y_test,predictions)
# print(acc)


glassClass={
    1 :"building_windows_float_processed",
    2:"building_windows_non_float_processed",
    3:"vehicle_windows_float_processed",
    4:"containers",
    5:"tableware",
    6:"headlamps"
}

NUM_CLASSES=len(glassClass.keys())
glassCSV=pd.read_csv("glass.csv",index_col=False)

X=glassCSV.drop(["Type"],axis=1)
X=X.drop(["RI"],axis=1)
Y=glassCSV["Type"]

#Y=pd.get_dummies(Y)

scaler=MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)

X=np.asarray(X)
Y=np.asarray(Y)
for i in range(0,len(Y)):
  if Y[i]>3:
    Y[i]-=1

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=100)
# X_train=np.asarray(X_train)
# Y_train=np.asarray(Y_train)

clf=RandomForest()
clf.fit(X_train,Y_train)

predictions=clf.predict(X_test)
acc= accuracy(Y_test,predictions)
print(acc)

# X_test=np.asarray(X_test)   
# Y_test=np.asarray(Y_test)

def predVals(testIdx,truthVal):
    X_t=np.expand_dims(X_test[testIdx],0)
    Y_T=glassClass[Y_test[testIdx]]
    pred=clf.predict(X_t)
    print(glassClass[np.argmax(pred)+1],Y_T )
    truthVal.append(np.argmax(pred)+1==Y_test[testIdx])
    return truthVal

truthVal=[]
for i in range(0,len(Y_test)):
    truthVal=predVals(i,truthVal)

for val in truthVal:
    print(val)