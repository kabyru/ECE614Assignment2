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
biasValues = [1e-5, 1e-4, 1e-3]
activityValues = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l1 = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l2 = [1e-5, 1e-4, 1e-3]        

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
                    for biasValue in biasValues:
                        for activityValue in activityValues:
                            for l1_l2_Value_l1 in l1_l2_Values_l1:
                                for l1_l2_Value_l2 in l1_l2_Values_l2:
                                    data.append([momentumValue, learningRateValue, architecture, numLayers, biasValue, activityValue, l1_l2_Value_l1, l1_l2_Value_l2])
    
    mp_handler(data)

