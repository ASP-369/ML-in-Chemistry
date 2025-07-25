def predVals(testIdx,truthVal):
    X_t=tf.expand_dims(X_test[testIdx],0)
    Y_T=glassClass[np.argmax(Y_test[testIdx])+1]
    pred=model.predict(X_t)  
    print(glassClass[np.argmax(pred)+1],Y_T )
    truthVal.append(np.argmax(pred)==np.argmax(Y_test[testIdx]))
    return truthVal
    

import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

Y=pd.get_dummies(Y)

scaler=MinMaxScaler()
scaler.fit(X)
scaler.transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=100)

tf.random.set_seed(42)
model=tf.keras.Sequential([
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")
 ])

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])

history=model.fit(X_train,Y_train,epochs=50,validation_split=0.33)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

X_test=np.asarray(X_test)
Y_test=np.asarray(Y_test)

truthVal=[]
for i in range(0,len(Y_test)):
    truthVal=predVals(i,truthVal)

for val in truthVal:
    print(val)

# composition=[]
# ele=["Na","Mg","Al","Si","K","Ca","Fe"]

# for i in range(0,7):
#     composition.append(input(f"Enter composition of:{ele[i]} "))

# X_t=tf.expand_dims(composition,0)
# print(X_t)
# pred=model.predict(X_t)  
# print(glassClass[np.argmax(pred)+1])