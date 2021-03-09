import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.datasets import fashion_mnist
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers

import csv

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

num_classes = 10
print('y data before: ')
print(y_train[0:5])

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)
print('\ny data after:')
print(y_train[0:5])

momentumValues = np.linspace(0,1,5)
learningRateValues = np.linspace(0,1,5)
architectureChoices = ['tanh','relu']
numLayersChoices = [1, 2, 3]
lossMeasure = 'categorical_crossentropy'
modelCounter = 0
biasValues = [1e-5, 1e-4, 1e-3]
activityValues = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l1 = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l2 = [1e-5, 1e-4, 1e-3]

with open('resultsRegularizers.csv', 'w', newline='') as csvfile:

    csvWriter = csv.writer(csvfile, delimiter=' ', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['MomentumRate', 'LearningRate', 'Architecture', 'NumberofLayers', 'biasValues', 'activityValues', 'kernelValueL1', 'kernelValueL2', 'ValidationAccuracy'])
    for momentumValue in momentumValues:
        for learningRateValue in learningRateValues:
            for architecture in architectureChoices:
                for numLayers in numLayersChoices:
                    for biasValue in biasValues:
                        for activityValue in activityValues:
                            for l1_l2_Value_l1 in l1_l2_Values_l1:
                                for l1_l2_Value_l2 in l1_l2_Values_l2:
                                    print("Now assessing Model #: " + str(modelCounter))
                                    if (numLayers == 1):
                                        model = Sequential()
                                        model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(10, activation='softmax'))
                                        sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
                                        model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
                                        model.summary()
                                    elif (numLayers == 2):
                                        model = Sequential()
                                        model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(10, activation='softmax'))
                                        sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
                                        model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
                                        model.summary()
                                    elif (numLayers == 3):
                                        model = Sequential()
                                        model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
                                        model.add(Dense(10, activation='softmax'))
                                        sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
                                        model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
                                        model.summary()
                                    
                                    training_samples = 60000
                                    testing_samples  = 10000

                                    batch_size = 128
                                    epochs     = 10

                                    history = model.fit(x_train[:training_samples],
                                                        y_train[:training_samples],
                                                        epochs=epochs,
                                                        batch_size=batch_size,
                                                        verbose=0,
                                                        validation_data=(x_test[:testing_samples],y_test[:testing_samples]))
                                    

                                    csvWriter.writerow([str(momentumValue), str(learningRateValue), architecture, str(numLayers), str(biasValue), str(activityValue), str(l1_l2_Value_l1), str(l1_l2_Value_l2), str(history.history['val_accuracy'][9])])
                                    modelCounter = modelCounter + 1