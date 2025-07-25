def predVals(testIdx,truthVal):
    X_t=np.asarray(tf.expand_dims(X_test[testIdx],0))
    Y_T=Y_test[testIdx]
    pred=model.predict(X_t)  
    truthVal.append(pred)
    return truthVal
    

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_transformer

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

X=glassCSV.drop(["RI"],axis=1)
Y=glassCSV["RI"]

ct= make_column_transformer(
    (MinMaxScaler(),["Na","Mg","Al","Si","K","Ca","Ba","Fe"])
)
ct.fit(X)
X_normal=ct.transform(X)

Y=np.asarray(Y)
X=np.asarray(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_normal,Y,test_size=0.20,random_state=100)
Y_train=tf.expand_dims(Y_train,-1)
Y_test=tf.expand_dims(Y_test,-1)

Y_train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)

tf.random.set_seed(42)
model=tf.keras.Sequential([
        tf.keras.layers.Dense(100,"relu"),
        tf.keras.layers.Dense(100,"relu"),
        tf.keras.layers.Dense(100,"relu"),
        tf.keras.layers.Dense(1)
 ])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])
history1=model.fit(X_train,Y_train,epochs=100,validation_split=0.33)

#pd.DataFrame(history1.history).plot()
plt.plot(history1.history["mae"])
plt.xlabel("loss")
plt.ylabel("epochs")


truthVal=[]
for i in range(0,len(Y_test)):
    truthVal=predVals(i,truthVal)

for i,val in enumerate(truthVal):
    print(val,Y_train[i])