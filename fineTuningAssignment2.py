from modelBuild import modelBuild
import multiprocessing
import numpy as np
import os

#data = {"momentumValue":momentumValue, "learningRateValue":learningRateValue, "architecture":architecture, "numLayers":numLayers}

momentumValues = np.linspace(0,1,10)
learningRateValues = np.linspace(0,1,10)
dropoutValues = np.linspace(0,0.75,10)
architectureChoices = ['sigmoid','tanh','relu']
numLayersChoices = [1, 2, 3]
optimizerChoices = ['sgd','RMSprop','Adam']
lossMeasure = 'categorical_crossentropy'

modelCounter = 0        

#print("MomentumRate " + "LearningRate " + "Architecture " + "NumberofLayers " + "ValidationAccuracy")

def mp_handler(data):
    p = multiprocessing.Pool(10)
    p.map(modelBuild, data)

if __name__ == '__main__':
    
    data = []
    
    for momentumValue in momentumValues:
        for learningRateValue in learningRateValues:
            for architecture in architectureChoices:
                for numLayers in numLayersChoices:
                    for optSelect in optimizerChoices:
                        for dropoutSelect in dropoutValues:
                            data.append([momentumValue, learningRateValue, architecture, numLayers, optSelect, dropoutSelect])
    
    mp_handler(data)

