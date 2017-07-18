import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('./data/driving_log.csv') as csvfile: #TODO check if .. or .
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/'+filename  #TODO check if .. or .
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


#Augment the data

# Keras NN
model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(X_train,y_train,validation_split = 0.2,shuffle=True,nb_epoch = 2)
model.save('model.h5')
