import keras 
from keras import *
from keras.datasets import *
from keras.layers import *
from keras.utils import *
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")
x_train/=255
x_test/=255

y_train=utils.to_categorical(y_train,10)
y_test=utils.to_categorical(y_test,10)
#print(x_train[45])
model=Sequential()
model.add(Dense(999,input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(55))

model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32)
