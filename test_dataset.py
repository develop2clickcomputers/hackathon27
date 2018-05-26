# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
attributes=9
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
print(dataset)

# load pima indians test dataset
dataset = numpy.loadtxt("pima-indians-diabetes-test.csv", delimiter=",")
# split into input (X) and output (Y) variables
attributeSize = dataset.size
if attributeSize  == attributes :X = dataset[0:8]
else:X = dataset[:,0:8]
print(X)