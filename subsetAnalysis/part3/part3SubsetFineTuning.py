from modelBuildRegularizers import modelBuild
import multiprocessing as mp
import time
import numpy as np
import os

part1TopTen = [[0.555555556,0.111111111,'tanh',2],
                [0.444444444,0.222222222,'tanh',1],
                [0.555555556,0.111111111,'relu',2],
                [0.555555556,0.111111111,'relu',3],
                [0.666666667,0.111111111,'tanh',3],
                [0.222222222,0.222222222,'tanh',3],
                [0.777777778,0.111111111,'tanh',2],
                [0,0.333333333,'tanh',2],
                [0.222222222,0.111111111,'tanh',2],
                [0.444444444,0.222222222,'tanh',2]]

lossMeasure = 'categorical_crossentropy'
biasValues = [1e-5, 1e-4, 1e-3]
activityValues = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l1 = [1e-5, 1e-4, 1e-3]
l1_l2_Values_l2 = [1e-5, 1e-4, 1e-3]

fn = './resultsRegularizersSubset.txt'

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
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(11)
    modelCounter = 0

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    data = []

    for subset in part1TopTen:
        momentumValue = subset[0]
        learningRateValue = subset[1]
        architecture = subset[2]
        numLayers = subset[3]
        for biasValue in biasValues:
            for activityValue in activityValues:
                for l1_l2_Value_l1 in l1_l2_Values_l1:
                    for l1_l2_Value_l2 in l1_l2_Values_l2:
                        data = [momentumValue, learningRateValue, architecture, numLayers, biasValue, activityValue, l1_l2_Value_l1, l1_l2_Value_l2]
                        job = pool.apply_async(modelBuild, (data, q))
                        jobs.append(job)
    
    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()
        #print(job.get())

        #DISPLAY COMPLETION PERCENTAGE
        modelCounter = modelCounter + 1
        progressPercent = (modelCounter / 810) * 100 #changed from 364000
        print("Models complete: " + str(modelCounter) + ", Percent Complete: " + str(progressPercent) + "%")
        #print(job.get())
    
    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()