from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import math
import time
from progress.bar import FillingSquaresBar
from queue import Queue
from multiprocessing import Process, Lock

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

bar = FillingSquaresBar('Training')

#values for quantization factor and truncation interval
Q = 8
T = 4

n1 = 0
n2 = 0

#number of iterations for the progress bar
numberIterations = 32
step = 5
currentIteration = 0

def QuantizeImage(img):
    #divide the image by Q
    #print(f'Quantizing image with Q = {Q}...')
    global bar
    bar = FillingSquaresBar('Training')
    bar.next(n=step)
    time.sleep(0.1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = img[i][j] / Q
    #print('Done quantizing the image!')
    #print('.', end='')
    bar.next(n=step)
    return img

def ComputeGradients(img):
    #print("Computing horizontal gradient (Gh)...")
    #print('.', end='')
    bar.next(n=step)
    global n1
    global n2
    n1, n2 = img.shape
    Gh = np.zeros([n1, n2], dtype=float)
    for i in range(0, n1):
        for j in range(0, n2 - 1):
            Gh[i][j] = float(img[i][j]) - float(img[i][j +1])
    #print('Computed horizontal gradient!')
    #print('.', end='')
    bar.next(n=step)

    #vertical gradient
    #print('Computing vertical gradient (Gv)...')
    Gv = np.zeros([n1, n2], dtype=float)
    for i in range(0, n1 -1):
        for j in range(0, n2):
            Gv[i][j] = float(img[i][j]) - float(img[i + 1][j])
    #print('Computed vertical gradient!')
    return [Gh, Gv]
    bar.next(n=step)

def TruncateGradients(Gh, Gv):
    #print(f'Truncating image gradients with T = {T}')
    #print("Truncating horizontal gradient with T")
    bar.next(n=step)
    for i in range(Gh.shape[0]):
        for j in range(Gh.shape[1]):
            if(Gh[i][j] > T):
                Gh[i][j] = T
            elif(Gh[i][j] < -T):
                Gh[i][j] = -T
    #print('Done truncating the horizontal gradient!')
    #print("Truncating vertical gradient with T")
    bar.next(n=step)
    for i in range(Gv.shape[0]):
        for j in range(Gv.shape[1]):
            if(Gv[i][j] > T):
                Gv[i][j] = T
            elif(Gv[i][j] < -T):
                Gv[i][j] = -T
    #print('Done truncating the vertical gradient!')
    bar.next(n=step)
    return [Gh, Gv]

def DetermineSecondOrderArrays(Gh, Gv):
    # determine second order co-occurrence array (horizontal)
    #print('Determining second order co-occurence arrays')
    #print('Determining horizontal SCAG...')
    bar.next(n=step)
    scagh = np.zeros([2 * T, 2 * T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0, 2 * T):
            s = i - T
            t = j - T
            sum = 0
            for k in range(0, n1):
                for l in range(0, n2 - 2):
                    if Gh[k][l] != s and Gh[k][l + 1] != t:
                        sum += 1
            scagh[i][j] = sum
    #print('Done determining horizontal SCAG!')
    bar.next(n=step)
    # print(scagh)
    # determine second order co-occurence array (vertical)
    #print('Determining vertical SCAG...')
    bar.next(n=step)
    scagv = np.zeros([2 * T, 2 * T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0, 2 * T):
            s = i - T
            t = j - T
            sum = 0
            for k in range(0, n1 - 2):
                for l in range(0, n2):
                    if Gv[k][l] != s and Gv[k + 1][l] != t:
                        sum += 1
            scagv[i][j] = sum
    #print('Done determining vertical SCAG!')
    bar.next(n=step)
    return [scagh, scagv]

def DetermineThirdOrderArrays(Gh, Gv):
    #print('Determining third-order co-occurence arrays on gradients')
    #print('Determining horizontal TCAG...')
    bar.next(n=step)
    tcagh = np.zeros([2 * T, 2 * T, 2 * T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0, 2 * T):
            for k in range(0, 2 * T):
                s = i - T
                t = j - T
                r = k - T
                sum = 0
                for l in range(0, n1):
                    for m in range(0, n2 - 3):
                        if Gh[l][m] != s and Gh[l][m + 1] != t and Gh[l][m + 2] != r:
                            sum += 1
                tcagh[i][j][k] = sum
    #print('Done determining horizontal TCAG!')
    bar.next(n=step)
    bar.next(n=step)
    # print(tcagh)
    #print('Determining vertical TCAG...')
    tcagv = np.zeros([2 * T, 2 * T, 2 * T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0, 2 * T):
            for k in range(0, 2 * T):
                s = i - T
                t = j - T
                r = k - T
                sum = 0
                for l in range(0, n1 - 3):
                    for m in range(0, n2):
                        if Gv[l][m] != s and Gv[l + 1][m] != t and Gv[l + 2][m] != r:
                            sum += 1
                tcagv[i][j][k] = sum
    # print(tcagv)
    bar.next(n=step)
    bar.next(n=step)
    #print('Done determining vertical TCAG!')
    return [tcagh, tcagh]

def NormalizeArrays(scagh, scagv, tcagh, tcagv):
    # Normalize the co-occurence array to eliminate the effect caused by the image size
    #print('Normalizing second-order co-occurence arrays')
    #print('Normalizing horizontal SCAG...')
    bar.next(n=step)
    # Determine the sum of all elements in the SCAG
    hsum2 = np.sum(scagh)
    # Normalize the scagh
    scagh = scagh / hsum2
    #print('Done normalizing horizontal SCAG')
    #print('Normalizing vertical SCAG...')
    bar.next(n=step)
    vsum2 = np.sum(scagv)
    scagv = scagv / vsum2
    #print('Done normalizing vertical SCAG')
    #print('Normalizing third-order co-occurence arrays')
    #print('Normalizing horizontal TCAG...')
    bar.next(n=step)
    hsum3 = np.sum(tcagh)
    tcagh = tcagh / hsum3
    #print('Done normalizing horizontal TCAG')
    #print('Normalizing vertical TCAG...')
    vsum3 = np.sum(tcagv)
    tcagv = tcagv / vsum3
    bar.next(n=step)
    #print('Done determining vertical tcag')
    return [scagh, scagv, tcagh, tcagv]

def ComputeFinalFeatures(imgFilename):
    #Read the image from file
    img = cv2.imread(imgFilename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = QuantizeImage(gray)
    [Gh, Gv] = ComputeGradients(gray)
    [Gh, Gv] = TruncateGradients(Gh, Gv)
    [scagh, scagv] = DetermineSecondOrderArrays(Gh, Gv)
    [tcagh, tcagv] = DetermineThirdOrderArrays(Gh, Gv)
    [scagh, scagv, tcagh, tcagv] = NormalizeArrays(scagh, scagv, tcagh, tcagv)
    tcag = (tcagh + tcagv) / 2
    #print('Final features:')
    bar.finish()
    return tcag


def TrainModel(svmFile=None):
    #Calculates a set of training final features which will be used to
    #train the svm to determine if the fingerprint is live or not
    #If the svmFile argument is not specified then the training has to be done
    # from the images in the Train folder
    # Images in good are live fingerprints and the ones in bad aren't

    #Take every image from the good folder
    goodDirectory = os.fsencode('Train/good')
    badDirectory = os.fsencode('Train/bad')
    #Clear the features file
    with open('Train/features.npy', 'w') as f:
        f.close()

    totalFiles = len(os.listdir(goodDirectory)) + len(os.listdir(badDirectory))

    features = 0
    classification = np.ndarray([totalFiles], dtype=int)
    print(f'Number files = {totalFiles}')
    currentIndex = 0
    for file in os.listdir(goodDirectory):
        print(f'\nCurrent file (good): {currentIndex + 1}/{totalFiles}')
        filename = os.fsdecode(file)
        #Compute the final value for the current file
        start = time.process_time()
        finalFeatures = ComputeFinalFeatures('Train/good/' + filename)
        end = time.process_time()
        print(f'\nElapsed time: ', end='')
        timer(start, end)
        #Append the data to the end of the finalFeatures.
        if currentIndex == 0:
            features = finalFeatures
        else:
            features = np.concatenate((features, finalFeatures))
        classification[currentIndex] = 0
        #Increment the current index
        currentIndex += 1

    #Take every image from the bad folder
    for file in os.listdir(badDirectory):
        print(f'Current file (bad): {currentIndex + 1}/{totalFiles}')
        filename = os.fsdecode(file)
        start = time.process_time()
        finalFeatures = ComputeFinalFeatures('Train/bad/' + filename)
        end = time.process_time()
        print(f'\nTime to process fingerprint: {end - start} seconds')
        features = np.concatenate((features, finalFeatures))
        classification[currentIndex] = 1
        currentIndex += 1

    #Write the results in file
    np.save('features.npy', features)
    np.save('classifications.npy', classification)

mutex = Lock()
featuresGlobal = 0
index = 0

def ProcessFile(queue, finished):
    counter = 0
    global index
    global featuresGlobal
    while True:
        if not queue.empty():
            filename = queue.get()
            #process the filename
            finalFeatures = ComputeFinalFeatures(filename)
            mutex.acquire()
            if index == 0:
                featuresGlobal = finalFeatures
            else:
                featuresGlobal = np.concatenate((featuresGlobal, finalFeatures))
            index += 1
            print(threading.current_thread() + 'finished file: ' + filename)
            mutex.release()
        else:
            q = finished.get()
            if q == True:
                break


import threading
from threading import Thread
def TrainModelMultithreaded():
    #Use a thread pool to train the model faster
    # Take every image from the good folder
    goodDirectory = os.fsencode('Train/good')
    badDirectory = os.fsencode('Train/bad')

    #Create the thread pool
    threads = []
    queue = Queue()
    finished = Queue()
    for i in range(4):
        threads.append(Thread(target=ProcessFile, args=[queue, finished], daemon=True))

    for thread in threads:
        thread.start()

    finished.put(False)
    for file in os.listdir(goodDirectory):
        filename = os.fsdecode(file)
        queue.put(filename)

    for file in os.listdir(badDirectory):
        filename = os.fsdecode(file)
        queue.put(filename)
    finished.put(True)

    for thread in threads:
        thread.join()

    print(featuresGlobal)

