import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from train import TrainModel
from train import TrainModelMultithreaded
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from future.moves import tkinter

def main():
    #TrainModel()

    #load from file
    classifications = np.load('classifications.npy')
    print(classifications)
    features = np.load('features.npy')

    #Create the correct shape array of the data
    features_good = 0
    for i in range(0, classifications.shape[0]):
        test1 = features[i * 8:(i+1)*8][:][:]
        test1 = test1.reshape(1, test1.shape[0] * test1.shape[1] * test1.shape[2])
        if i == 0:
            features_good = test1
        else:
            features_good = np.concatenate((features_good, test1))

    print(features_good.shape)
    scaler = StandardScaler()
    features_good = scaler.fit_transform(features_good)
    clf = svm.SVC(probability=True)
    clf.fit(features_good, classifications)

    #TrainModelMultithreaded()
    #read the fingerprint image
    img = cv2.imread('Database1/101_1.tif')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #n1 and n2 are the height and width of a gray image
    [n1, n2] = gray.shape
    print('Dimensiune imagine = ', n1, n2)
    #cv2.imshow('Grayscale', gray)
    # waiting until key press
    #cv2.waitKey()

    #pick a Q which will be the quantization factor (Q >= 1)
    Q = 16
    #divide the image by Q
    print(f'Quantizing image with Q = {Q}...')
    for i in range(n1):
        for j in range(n2):
            gray[i][j] = gray[i][j] / Q
    print('Done quantizing the image!')

    #cv2.imshow('Grayscale 2', gray)
    #cv2.waitKey()

    #compute image gradients
    #horizontal gradient
    print("Computing horizontal gradient (Gh)...")
    Gh = np.zeros([n1, n2], dtype=float)
    for i in range(0, n1):
        for j in range(0, n2 - 1):
            Gh[i][j] = float(gray[i][j]) - float(gray[i][j +1])
            #print(float(gray[i][j]) - float(gray[i][j +1]))
    print('Computed horizontal gradient!')

    #vertical gradient
    print('Computing vertical gradient (Gv)...')
    Gv = np.zeros([n1, n2], dtype=float)
    for i in range(0, n1 -1):
        for j in range(0, n2):
            Gv[i][j] = float(gray[i][j]) - float(gray[i+1][j])
    print('Computed vertical gradient!')
    #set the value of T
    T = 4
    #truncate the gradients
    print(f'Truncating image gradients with T = {T}')
    print("Truncating horizontal gradient with T")
    for i in range(Gh.shape[0]):
        for j in range(Gh.shape[1]):
            if(Gh[i][j] > T):
                Gh[i][j] = T
            elif(Gh[i][j] < -T):
                Gh[i][j] = -T
    print('Done truncating the horizontal gradient!')
    print("Truncating vertical gradient with T")
    for i in range(Gv.shape[0]):
        for j in range(Gv.shape[1]):
            if(Gv[i][j] > T):
                Gv[i][j] = T
            elif(Gv[i][j] < -T):
                Gv[i][j] = -T
    print('Done truncating the vertical gradient!')

    #determine second order co-occurrence array (horizontal)
    print('Determining second order co-occurence arrays')
    print('Determining horizontal SCAG...')
    scagh = np.zeros([2 * T, 2 * T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0, 2 * T):
            s = i - T
            t = j - T
            sum = 0
            for k in range(0, n1):
                for l in range(0, n2 -2):
                    if Gh[k][l] != s and Gh[k][l + 1] != t:
                        sum += 1
            scagh[i][j] = sum
    print('Done determining horizontal SCAG!')

    #print(scagh)
    #determine second order co-occurence array (vertical)
    print('Determining vertical SCAG...')
    scagv = np.zeros([2*T, 2*T], dtype=float)
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
    print('Done determining vertical SCAG!')

    #print(scagv)
    print('Determining third-order co-occurence arrays on gradients')
    print('Determining horizontal TCAG...')
    tcagh = np.zeros([2*T, 2*T, 2*T], dtype=float)
    for i in range(0, 2 * T):
        for j in range(0,2*T):
            for k in range(0, 2 *T):
                s = i - T
                t = j - T
                r = k - T
                sum = 0
                for l in range(0, n1):
                    for m in range(0, n2 - 3):
                        if Gh[l][m] != s and Gh[l][m + 1] != t and Gh[l][m + 2] != r:
                            sum += 1
                tcagh[i][j][k] = sum
    print('Done determining horizontal TCAG!')
    #print(tcagh)
    print('Determining vertical TCAG...')
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
    #print(tcagv)
    print('Done determining vertical TCAG!')
    #print("ended!")

    #Normalize the co-occurence array to eliminate the effect caused by the image size
    print('Normalizing second-order co-occurence arrays')
    print('Normalizing horizontal SCAG...')
    #Determine the sum of all elements in the SCAG
    hsum2 = np.sum(scagh)
    #Normalize the scagh
    scagh = scagh / hsum2
    print('Done normalizing horizontal SCAG')
    print('Normalizing vertical SCAG...')
    vsum2 = np.sum(scagv)
    scagv = scagv / vsum2
    print('Done normalizing vertical SCAG')
    print('Normalizing third-order co-occurence arrays')
    print('Normalizing horizontal TCAG...')
    hsum3 = np.sum(tcagh)
    tcagh = tcagh / hsum3
    print('Done normalizing horizontal TCAG')
    print('Normalizing vertical TCAG...')
    vsum3 = np.sum(tcagv)
    tcagv = tcagv / vsum3
    print('Done determining vertical tcag')
    #Determine the final features
    print('Computing final features...')
    tcag = (tcagh + tcagv) / 2
    print('Final features:')

    test_data = tcag.reshape(1, tcag.shape[0] * tcag.shape[1] * tcag.shape[2])
    test_data = scaler.fit_transform(test_data)
    y_prob_pred = clf.predict_proba(test_data)
    print(y_prob_pred)

    infoLive = 'Live: ' + str("{:.2f}".format(y_prob_pred[0][0]))
    infoNotLive = ' Not live: ' + str("{:.2f}".format(y_prob_pred[0][1]))

    imgplot = plt.imshow(img)
    plt.title('Fingerprint')
    plt.text(1, -30, infoLive, fontsize=20, color='red')
    plt.text(1, 350, infoNotLive, fontsize=20, color='red')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
