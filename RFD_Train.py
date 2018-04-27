import numpy as np #import numpy library for operation on matrix and array
import cv2 as openCV #import openCV library
from skimage import feature #import scikit-image for calculating hog feature
from RFD_Utils import cropImg #import other functions
from sklearn.decomposition import PCA #import this library for calculating PCA on hog features
import skimage
from skimage import img_as_float #import that for converting image from openCV to scikit
from skimage import img_as_ubyte #import that for converting image from scikit to openCV

def train():

    nCams = 3 #number of photocameras
    nPhoto = 9 #number of photos of each camera
    diamParameter = 8 #diam_parameter for creating morphological element as disk with diameter of 15px
    maxThrCanny = 150 #max threshold for appling canny algorithm on each image
    minThrCanny = maxThrCanny * 0.04 #min threshold for appling canny algorithm on each image
    
    numStrongest = 20 #number of strongest points
    numHog = 16 #number of hog features values for each point
    
    RFD_Diam_Limit = np.zeros((nCams,2))
    RFD_HOG_Limit = np.zeros((nCams,2))
    RFD_HOG_Mu = np.zeros((nCams,numHog))
    RFD_HOG_Coeff = np.zeros((nCams ,numHog))
        
    #margin used to determinate threshold of derivate of diamater and threshold of mean of hog features
    diamMarg = 3.5
    hogMarg = 10
    
    #dimension of mobile window used to calculate derivate of diameter
    window = 150
    guard = 15
    crop = window + guard
    
    nC = np.shape(openCV.imread("TrainingSet/Cam"+str(1)+"/"+str(2)+".png"))[1] #num columns of each image

    print("TrainingSet")
    #processing each image of each camera
    for cc in range(1, nCams + 1, 1):
        print("Cam: " + str(cc))
        hog = np.zeros((1,numHog)) #declare vector for saving hog features values of each strongest point
        diffDiameter = np.zeros((nC - crop * 2, nPhoto))#declare vector for saving derivate of diamater values
        for pp in range(2, nPhoto + 2, 1):
            print("Photo: " + str(pp))
            img = openCV.imread("TrainingSet/Cam"+str(cc)+"/"+str(pp)+".png") #read image
            disk = openCV.getStructuringElement(openCV.MORPH_ELLIPSE, (2 * diamParameter - 1, 2 * diamParameter - 1))  # 15x15 matrix for closing matrix and obtaining a BW img
            imgGray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY)  # convert image from bgr to gray scale
            c = openCV.Canny(imgGray, maxThrCanny, minThrCanny) #apply canny algotithm to grayscale image
            f = openCV.morphologyEx(c, openCV.MORPH_CLOSE, disk)  # f = BW img - close image --> img_as_ubyte convert scikit to openCV 
            blank_image, bH, bL = cropImg(f, img) #crop image
            
            diameter = np.subtract(bH, bL)#calculate diamater
            
            diff = np.gradient(diameter, axis=0) # calculate derivate
            """
            delete first and last element because At the boundaries, the first difference is calculated. 
            This means that at each end of the array, the gradient given is simply, the difference between the end two values (divided by 1)
            Away from the boundaries the gradient for a particular index is given by taking the difference between the the values either side and dividing by 2.
            """
            diff = np.delete(diff, 0)
            diff = np.delete(diff, np.shape(diff)[0] - 1)

            #calculate mean of derivate of each mobile window
            for ii in range(crop, nC - crop, 1):
                if ii <= window:
                    diffDiameter[ii - crop, pp - 2] = np.mean(diff[0 : window, 0])
                elif ii >= (nC - window) :
                    diffDiameter[ii - crop, pp - 2] = np.mean(diff[np.shape(diff)[0] - window : np.shape(diff)[0], 0])
                else:
                    diffDiameter[ii - crop, pp - 2] = np.mean(diff[ii - window : ii + window])
            
            #detect fast features 
            blank_image = openCV.cvtColor(blank_image, openCV.COLOR_BGR2RGB) #convert image from bgr to rgb scale
            imgHisteqClean = skimage.exposure.equalize_hist(skimage.color.rgb2gray(blank_image), nbins=64)#increase contrast of image
            fast = openCV.FastFeatureDetector_create(90) #apply FAST algorithm on equalized image
            imgHisteqClean = img_as_ubyte(imgHisteqClean) #convert image for using it with to opencv library
            kp = fast.detect(imgHisteqClean)#detect corners with FAST algorithm
            
            #select 20 strongest points-- bubblesort
            numKP = len(kp)
            strongestPt = np.zeros((numStrongest,2))

            for i in range(0, numKP - 1, 1):
                for j in range(i + 1, numKP, 1):
                    if(kp[j].response > kp[i].response):
                        t = kp[i]
                        kp[i] = kp[j]
                        kp[j] = t

            #consider only first 20 points that were ordinated before
            for i in range(0, numStrongest, 1):
                strongestPt[i][0] = int(kp[i].pt[0])
                strongestPt[i][1] = int(kp[i].pt[1])

            #HOG Features
            vect = np.zeros((1,numHog))
            incr = 50
            #for each "strongest point" create a 100x100 sub-image where "strongest point" is in the middle of image and calculate hog features value
            for i in range(0, numStrongest, 1):
                if(int(strongestPt[i][1]) > incr and int(strongestPt[i][0] > incr)):
                    limRowU = int(strongestPt[i][1]) - incr
                    limRowD = int(strongestPt[i][1]) + incr
                    limColL = int(strongestPt[i][0]) - incr
                    limColR = int(strongestPt[i][0]) + incr
                    roi = imgHisteqClean[limRowU : limRowD, limColL : limColR]
                    #divide 100x100 sub-image in cells and consider two blocks of cells each time. Cells are 50x50
                    (H, hogImage) = feature.hog(roi, orientations=4, pixels_per_cell=(50,50), cells_per_block=(2, 2), transform_sqrt=True, visualise=True, block_norm='L2-Hys', feature_vector=True)
                    if len(H) == 16:
                        for j in range(0, np.shape(H)[0], 1):
                            vect[0,j] = H[j]
                    
                        hog = np.concatenate((hog, vect), axis=0)
            
        hog = np.delete(hog, 0, 0)
        #mean of mean of each column
        mD = np.mean(np.mean(diffDiameter))
        #mean of standar deviation of each column
        sD = np.mean(np.std(diffDiameter, ddof=1))
        
        #diameter threshold
        RFD_Diam_Limit[cc - 1, 0] = np.around(mD + diamMarg * sD, decimals=4)
        RFD_Diam_Limit[cc - 1, 1] = np.around(mD - diamMarg * sD, decimals=4)
        
        #calculate PCA

        meanCols = np.zeros((numHog,1))
        for i in range(0, np.shape(hog)[0], 1):
            for j in range(0, np.shape(hog)[1],1):
                meanCols[j] = np.mean(hog[:,j])
                RFD_HOG_Mu[cc - 1, j] = meanCols[j] #mean of each column of hog features matrix
    
        D = np.zeros((np.shape(hog)[0], np.shape(hog)[1]))
        
        for i in range(0, np.shape(hog)[0], 1):
            for j in range(0, np.shape(hog)[1],1):
                D[i,j] = hog[i,j] - meanCols[j]
                
        eps = (1/(np.shape(D)[0] - 1)) * np.dot(D.T, D)#covar matrix
        
        #PCA
        pca = PCA(n_components=1)
        pca = pca.fit(hog)

        RFD_HOG_Coeff[cc - 1] = pca.components_ #eigvector
        components = RFD_HOG_Coeff[cc - 1]
        score = np.dot(D, components) #projected values
        
        #HOG Threshold

        mH = np.mean(score)
        sH = np.std(score, ddof=1)
        
        mH = np.mean(score)
        sH = np.std(score, ddof=1)
        tau = hogMarg * (sH/(np.sqrt(len(score))))
        
        RFD_HOG_Limit[cc - 1, 0] = mH + 1  * tau
        RFD_HOG_Limit[cc - 1, 1] = mH - 1  * tau 
        
        print("-----")
        print("DIAM. PARAMETER:")
        print("Cam: " + str(cc) + " - RFD_DiamLimitUP: " + str(RFD_Diam_Limit[cc - 1, 0]))
        print("Cam: " + str(cc) + " - RFD_DiamLimitDOWN: " + str(RFD_Diam_Limit[cc - 1, 1]))
        print("Cam: " + str(cc) + " - mD: " + str(np.around(mD,decimals=4)))
        print("Cam: " + str(cc) + " - sD: " + str(np.around(sD, decimals=4)))
        print("HOG PARAMETER:")
        print("Cam: " + str(cc) + " - RFD_par.HOG_mu: " + str(np.around(np.mean(RFD_HOG_Mu[cc - 1]), decimals=4)))
        print("Cam: " + str(cc) + " - RFD_par.HOG_coeff: " + str(np.around(np.mean(RFD_HOG_Coeff[cc - 1]), decimals=4)))
        print("Cam: " + str(cc) + " - mH: " + str(mH))
        print("Cam: " + str(cc) + " - sH: " + str(sH))
        print("Cam: " + str(cc) + " - tau: " + str(tau))
        print("Cam: " + str(cc) + " - RFD_HOG_LimitUP: " + str(RFD_HOG_Limit[cc - 1, 0]))
        print("Cam: " + str(cc) + " - RFD_HOG_LimitDOWN: " + str(RFD_HOG_Limit[cc - 1, 1]))
        print("-----")

    return RFD_Diam_Limit,RFD_HOG_Mu,RFD_HOG_Coeff,RFD_HOG_Limit
    
      
    
    
    