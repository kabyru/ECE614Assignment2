from modelBuildRegularizersOptimizersDropout import modelBuild
import multiprocessing as mp
import time
import numpy as np
import os

#data = {"momentumValue":momentumValue, "learningRateValue":learningRateValue, "architecture":architecture, "numLayers":numLayers}

momentumValue = 0.555555555555555
learningRateValue = 0.111111111111111
architecture = 'tanh'
numLayers = 2
lossMeasure = 'categorical_crossentropy'
modelCounter = 0
biasValue = 0.0001
activityValue = 0.00001
l1_l2_Value_l1 = 0.00001
l1_l2_Value_l2 = 0.00001
optChoices = ['sgd','Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'] #removed RMSprop
dropoutValues = np.linspace(0,0.8,10) #changed from 1,0.75,10

#Progress Bar ezpz code
#364500 max iterations
#Let's count how many iterations have ran and act accordingly
 

fn = './resultsOptimizers.txt'

#momentumValues = np.linspace(0,1,2)
#learningRateValues = np.linspace(0,1,2)
#architectureChoices = ['relu']
#numLayersChoices = [3]
#lossMeasure = 'categorical_crossentropy'
#modelCounter = 0         

#print("MomentumRate " + "LearningRate " + "Architecture " + "NumberofLayers " + "ValidationAccuracy")

def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open(fn, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()

def main():
    modelCounter = 0
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(11)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    data = []
    
    startTime = time.time()
    for optSelect in optChoices:
        for dropoutValue in dropoutValues:
            data = [momentumValue, learningRateValue, architecture, numLayers, biasValue, activityValue, l1_l2_Value_l1, l1_l2_Value_l2, optSelect, dropoutValue]
            job = pool.apply_async(modelBuild, (data, q))
            jobs.append(job)
    
    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

        #DISPLAY COMPLETION PERCENTAGE

        modelCounter = modelCounter + 1
        progressPercent = (modelCounter / 80) * 100 #changed from 364000
        print("Models complete: " + str(modelCounter) + ", Percent Complete: " + str(progressPercent) + "%")
        #print(job.get())
    
    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
