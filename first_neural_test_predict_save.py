# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
# fix random seed for reproducibility
seed = 7
attributes=9
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#--print(dataset)
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
   json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# load pima indians dataset
dataset= numpy.loadtxt("pima-indians-diabetes-test.csv", delimiter=",")
# split into input (X) and output (Y) variables
attributeSize = dataset.size
print(attributeSize)
if attributeSize  == attributes :X = dataset[0:8]
else:X = dataset[:,0:8]
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(X[0]) for X  in predictions]
print(rounded)