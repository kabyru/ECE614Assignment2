from modelBuildRegularizersOptimizersDropout import modelBuild
import multiprocessing as mp
import time
import numpy as np
import os

part3TopFive = [[0.555555556,0.111111111,'relu',2,0.0001,0.0001,1.00E-05,0.0001],[0.444444444,0.222222222,'tanh',2,0.0001,1.00E-05,1.00E-05,1.00E-05],[0.555555556,0.111111111,'relu',2,0.001,0.0001,1.00E-05,1.00E-05],[0.555555556,0.111111111,'relu',3,0.0001,1.00E-05,1.00E-05,1.00E-05],[0,0.333333333,'tanh',2,0.001,0.0001,1.00E-05,1.00E-05]]

lossMeasure = 'categorical_crossentropy'
optChoices = ['sgd','Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
dropoutValues = np.linspace(0,0.8,10) #changed from 1,0.75,10

fn = './resultsOptimizersSubset.txt'

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
    for subset in part3TopFive:
        momentumValue = subset[0]
        learningRateValue = subset[1]
        architecture = subset[2]
        numLayers = subset[3]
        biasValue = subset[4]
        activityValue = subset[5]
        l1_l2_Value_l1 = subset[6]
        l1_l2_Value_l2 = subset[7]
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
        progressPercent = (modelCounter / 400) * 100 #changed from 364000
        print("Models complete: " + str(modelCounter) + ", Percent Complete: " + str(progressPercent) + "%")
        #print(job.get())
    
    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
