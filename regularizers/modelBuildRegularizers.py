def modelBuild(data, q):
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

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train/255
    x_test = x_test/255

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    #data.append([momentumValue, learningRateValue, architecture, numLayers])
    momentumValue = data[0]
    learningRateValue = data[1]
    architecture = data[2]
    numLayers = data[3]
    #data.append([momentumValue, learningRateValue, architecture, numLayers, biasValue, activityValue, l1_l2_Value_l1, l1_l2_Value_l2])
    biasValue = data[4]
    activityValue = data[5]
    l1_l2_Value_l1 = data[6]
    l1_l2_Value_l2 = data[7]
    lossMeasure = 'categorical_crossentropy'

    try:
        if (numLayers == 1):
            model = Sequential()
            model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(10, activation='softmax'))
            sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
            model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
            #model.summary()
        elif (numLayers == 2):
            model = Sequential()
            model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(10, activation='softmax'))
            sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
            model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
            #model.summary()
        elif (numLayers == 3):
            model = Sequential()
            model.add(Dense(64, activation=architecture, input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(64, activation=architecture, kernel_regularizer=regularizers.l1_l2(l1=l1_l2_Value_l1, l2=l1_l2_Value_l2), bias_regularizer=regularizers.l2(biasValue), activity_regularizer=regularizers.l2(activityValue)))
            model.add(Dense(10, activation='softmax'))
            sgd = optimizers.SGD(lr=learningRateValue, decay=0, momentum=momentumValue)
            model.compile(loss=lossMeasure, optimizer=sgd, metrics=['accuracy'])
            #model.summary()
                                    
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
                                    
        #print(str(momentumValue) + " " + str(learningRateValue) + " " + architecture + " " + str(numLayers) + " " + str(biasValue) + " " + str(activityValue) + " " + str(l1_l2_Value_l1) + " " + str(l1_l2_Value_l2) + " " + str(history.history['val_accuracy'][9]))
        res = str(momentumValue) + " " + str(learningRateValue) + " " + architecture + " " + str(numLayers) + " " + str(biasValue) + " " + str(activityValue) + " " + str(l1_l2_Value_l1) + " " + str(l1_l2_Value_l2) + " " + str(history.history['val_accuracy'][9])
        q.put(res)
        return res

    except Exception:
        import traceback
        print(traceback.format_exc())
    
    