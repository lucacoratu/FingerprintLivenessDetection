import concurrent.futures

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
from sklearn.preprocessing import StandardScaler

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
prefix = 'Training'

class LivenessDetectionModel:
    features = 0
    classifications = 0
    loaded = False
    testingDirectory = None
    wrongPredicitons = []

    def __init__(self):
        pass

    def QuantizeImage(self, img):
        #divide the image by Q
        #print(f'Quantizing image with Q = {Q}...')
        global bar
        bar = FillingSquaresBar(prefix)
        bar.next(n=step)
        time.sleep(0.1)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = img[i][j] / Q
        #print('Done quantizing the image!')
        #print('.', end='')
        bar.next(n=step)
        return img

    def ComputeGradients(self, img):
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
        bar.next(n=step)
        return [Gh, Gv]


    def TruncateGradients(self, Gh, Gv):
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

    def DetermineSecondOrderHorizontal(self, Gh):
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
        return Gh

    def DetermineSecondOrderVertical(self, Gv):
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
        return scagv

    def DetermineSecondOrderArrays(self, Gh, Gv):
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     f1 = executor.submit(DetermineSecondOrderHorizontal, Gh)
        #     f2 = executor.submit(DetermineSecondOrderVertical, Gv)
        #     scagh = f1.result()
        #     scagv = f2.result()
        scagh = self.DetermineSecondOrderHorizontal(Gh)
        scagv = self.DetermineSecondOrderVertical(Gv)

        return [scagh, scagv]

    def DetermineThirdOrderHorizontal(self, Gh):
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
        return tcagh

    def DetermineThirdOrderVertical(self, Gv):
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
        return tcagv

    def DetermineThirdOrderArrays(self, Gh, Gv):
        tcagh = self.DetermineThirdOrderHorizontal(Gh)
        tcagv = self.DetermineThirdOrderVertical(Gv)

        return [tcagh, tcagv]

    def NormalizeArrays(self, scagh, scagv, tcagh, tcagv):
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

    def ComputeFinalFeatures(self, imgFilename):
        #Read the image from file
        img = cv2.imread(imgFilename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #scale the image down
        gray = cv2.resize(gray, (64, 64))
        gray = self.QuantizeImage(gray)
        [Gh, Gv] = self.ComputeGradients(gray)
        [Gh, Gv] = self.TruncateGradients(Gh, Gv)
        [scagh, scagv] = self.DetermineSecondOrderArrays(Gh, Gv)
        [tcagh, tcagv] = self.DetermineThirdOrderArrays(Gh, Gv)
        [scagh, scagv, tcagh, tcagv] = self.NormalizeArrays(scagh, scagv, tcagh, tcagv)
        tcag = (tcagh + tcagv) / 2
        #print('Final features:')
        bar.finish()
        return tcag


    def TrainModel(self, svmFile=None):
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
            finalFeatures = self.ComputeFinalFeatures('Train/good/' + filename)
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
            finalFeatures = self.ComputeFinalFeatures('Train/bad/' + filename)
            end = time.process_time()
            print(f'\nTime to process fingerprint: {end - start} seconds')
            features = np.concatenate((features, finalFeatures))
            classification[currentIndex] = 1
            currentIndex += 1

        #Write the results in file
        np.save('features.npy', features)
        self.features = features
        np.save('classifications.npy', classification)
        self.classifications = classification

    def LoadModel(self, featuresFilename, classificationFilename):
        self.classifications = np.load(classificationFilename)
        self.features = np.load(featuresFilename)
        self.loaded = True
        print('Model has been loaded from file!')

    def TestModel(self, directory='Test/'):
        #Test the model with all the files inside the Test directory
        if self.loaded == False:
            raise Exception('You have to load the model before testing!')

        # Create the correct shape array of the data
        features_good = 0
        for i in range(0, self.classifications.shape[0]):
            test1 = self.features[i * 2 * T:(i + 1) * 2 * T][:][:]
            test1 = test1.reshape(1, test1.shape[0] * test1.shape[1] * test1.shape[2])
            if i == 0:
                features_good = test1
            else:
                features_good = np.concatenate((features_good, test1))

        #print(features_good.shape)
        # Train svm using test data
        scaler = StandardScaler()
        features_good = scaler.fit_transform(features_good)
        clf = svm.SVC(probability=True, kernel='rbf')
        clf.fit(features_good, self.classifications)
        print('SVC is prepared for testing!')

        global prefix
        prefix = 'Testing'

        # For every file in good and bad directory inside Test directory
        # take the photo and use the model to predict the liveness
        goodDirectory = os.fsencode(directory + 'good')
        badDirectory = os.fsencode(directory + 'bad')
        totalFiles = len(os.listdir(goodDirectory)) + len(os.listdir(badDirectory))

        classification_test = np.ndarray([totalFiles], dtype=int)
        print(f'Number files for testing = {totalFiles}')
        currentIndex = 0
        test_features = 0
        for file in os.listdir(goodDirectory):
            print(f'\nTesting file : {currentIndex + 1}/{totalFiles}')
            filename = os.fsdecode(file)
            # Compute the final value for the current file
            start = time.process_time()
            finalFeatures = self.ComputeFinalFeatures(directory + 'good/' + filename)
            end = time.process_time()
            print(f'\nElapsed time: ', end='')
            timer(start, end)
            if currentIndex == 0:
                test_features = finalFeatures
            else:
                test_features = np.concatenate((test_features, finalFeatures))
            classification_test[currentIndex] = 0
            # Increment the current index
            currentIndex += 1

        # Take every image from the bad folder
        for file in os.listdir(badDirectory):
            print(f'Testing file : {currentIndex + 1}/{totalFiles}')
            filename = os.fsdecode(file)
            start = time.process_time()
            finalFeatures = self.ComputeFinalFeatures(directory + 'bad/' + filename)
            end = time.process_time()
            print(f'\nElapsed time: ', end='')
            timer(start, end)
            test_features = np.concatenate((test_features, finalFeatures))
            classification_test[currentIndex] = 1
            currentIndex += 1

        #Give the test features to the model and save the probabilities after
        # Create the correct shape array of the data
        features_test_good = 0
        print(classification_test.shape)
        for i in range(0, classification_test.shape[0]):
            test1 = test_features[i * 2 * T:(i + 1) * 2 * T][:][:]
            test1 = test1.reshape(1, test1.shape[0] * test1.shape[1] * test1.shape[2])
            if i == 0:
                features_test_good = test1
            else:
                features_test_good = np.concatenate((features_test_good, test1))

        scaler = StandardScaler()
        features_test_good = scaler.fit_transform(features_test_good)
        y_pred_proba = clf.predict_proba(features_test_good)

        np.save('testproba.npy', y_pred_proba)
        print('Saved the probabilities in the testproba.npy file!')
        print(y_pred_proba)

        # for i in range(classification_test.shape[0]):
        #     if classification_test[i] == 0:
        #         classification_test[i] = 1
        #     else:
        #         classification_test[i] = 0

        #Make a statistic of the test
        rightPredictions = []
        wrongPredictions = []
        inconcludentPredictions = []
        for i in range(y_pred_proba.shape[0]):
            #Check if the guess was right or wrong
            diff = abs(y_pred_proba[i][0] - y_pred_proba[i][1])
            if diff <= 0.1:
                #Probabilities are too close to each other
                inconcludentPredictions.append(i)
            else:
                if y_pred_proba[i][0] > y_pred_proba[i][1] and classification_test[i] == 0:
                    rightPredictions.append(i)
                elif y_pred_proba[i][0] < y_pred_proba[i][1] and classification_test[i] == 1:
                    rightPredictions.append(i)
                else:
                    wrongPredictions.append(i)

        print(f'Right predicitons of the algorithm ({len(rightPredictions)}): {rightPredictions}')
        print(f'Wrong predictions of the algorithm ({len(wrongPredictions)}): {wrongPredictions}')
        print(f'Inconcludent predicitons of the alogrithm ({len(inconcludentPredictions)}): {inconcludentPredictions}')
        print(f'Right prediction probability: {len(rightPredictions) / totalFiles}')
        self.wrongPredicitons = wrongPredictions
        self.testingDirectory = directory

    def PlotWrongPredictions(self):
        #Go through the files in the testing directory and take only the ones where the current
        #index is equal to a value in wrongPredictions

        goodDirectory = os.fsencode(self.testingDirectory + 'good')
        badDirectory = os.fsencode(self.testingDirectory + 'bad')
        totalFiles = len(os.listdir(goodDirectory)) + len(os.listdir(badDirectory))

        currentIndex = 0
        images = []
        for file in os.listdir(goodDirectory):
            #Check if this file appears in
            if self.wrongPredicitons.count(currentIndex) != 0:
                filename = os.fsdecode(file)
                img = cv2.imread(self.testingDirectory + '/good/' + filename)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
            currentIndex += 1

        for file in os.listdir(badDirectory):
            #Check if this file appears in
            if self.wrongPredicitons.count(currentIndex) != 0:
                filename = os.fsdecode(file)
                img = cv2.imread(self.testingDirectory + '/bad/' + filename)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
            currentIndex += 1

        plt.gcf().set_size_inches(15,15)
        for i in range(len(images)):
            #Append the image to the
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i], cmap='gray')

        plt.waitforbuttonpress()

