from modelBuild import modelBuild
import multiprocessing
import numpy as np
import os

#data = {"momentumValue":momentumValue, "learningRateValue":learningRateValue, "architecture":architecture, "numLayers":numLayers}

momentumValues = np.linspace(0,1,10)
learningRateValues = np.linspace(0,1,10)
architectureChoices = ['sigmoid','tanh','relu']
numLayersChoices = [1, 2, 3]
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
                    data.append([momentumValue, learningRateValue, architecture, numLayers])
    
    mp_handler(data)

