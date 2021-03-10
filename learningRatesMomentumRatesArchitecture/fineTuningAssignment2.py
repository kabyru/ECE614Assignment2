from modelBuild import modelBuild
import multiprocessing as mp
import time
import numpy as np
import os

#data = {"momentumValue":momentumValue, "learningRateValue":learningRateValue, "architecture":architecture, "numLayers":numLayers}

momentumValues = np.linspace(0,1,10)
learningRateValues = np.linspace(0,1,10)
architectureChoices = ['tanh','relu']
numLayersChoices = [1, 2, 3]
lossMeasure = 'categorical_crossentropy'
modelCounter = 0
 

fn = './resultsPart1.txt' 

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

    for momentumValue in momentumValues:
        for learningRateValue in learningRateValues:
            for architecture in architectureChoices:
                for numLayers in numLayersChoices:
                    data = [momentumValue, learningRateValue, architecture, numLayers]
                    job = pool.apply_async(modelBuild, (data, q))
                    jobs.append(job)
    
    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

        #DISPLAY COMPLETION PERCENTAGE
        modelCounter = modelCounter + 1
        progressPercent = (modelCounter / 600) * 100 #changed from 364000
        print("Models complete: " + str(modelCounter) + ", Percent Complete: " + str(progressPercent) + "%")
        #print(job.get())
    
    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
