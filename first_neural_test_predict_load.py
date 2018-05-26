# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
# fix random seed for reproducibility
seed = 7
attributes=9
# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
dataset = numpy.loadtxt("pima-indians-diabetes-test.csv", delimiter=",")
# split into input (X) and output (Y) variables
attributeSize = dataset.size
print(attributeSize)
if attributeSize  == attributes :X = dataset[0:8]
else:X = dataset[:,0:8]
# calculate predictions
predictions = loaded_model.predict(X)
# round predictions
rounded = [round(X[0]) for X  in predictions]
print(rounded)