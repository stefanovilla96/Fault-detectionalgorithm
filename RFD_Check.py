import numpy as np #import numpy library for operation on matrix and array
import cv2 as openCV #import openCV library
from skimage import feature #import scikit-image for calculating hog feature
from RFD_Utils import cropImg,drawPlotDiameter,drawPlotProjection,drawTable, drawPlotDiameterHealthy, drawPlotProjectionHealthy #import other functions for draw plot and table
from sklearn.decomposition import PCA #import this library for calculating PCA on hog features
import skimage
from skimage import img_as_float #import that for converting image from openCV to scikit
from skimage import img_as_ubyte #import that for converting image from scikit to openCV

def check(diamLimit, HOGLimit, HOGMu, HOGCoeff):

    print("FaultCase")
    nCams = 3 #number of photocameras
    diamParameter = 8 #diam_parameter for creating morphological element as disk with diameter of 15
    maxThrCanny = 150 #max threshold for appling canny algorithm on each image
    minThrCanny = maxThrCanny * 0.04 #min threshold for appling canny algorithm on each image
 
    #dimension of mobile window used to calculate derivate of diameter
    window = 150
    guard = 15
    crop = window + guard
    
    numStrongest = 20 #number of strongest points
    numHog = 16 #number of hog features values for each point

    nC = np.shape(openCV.imread("FaultCase/Cam"+str(1)+"/"+str(1)+".png"))[1] #num columns of each image
    
    #declare this two vectors for print final result
    diamCheck = np.zeros((nCams, 1), dtype=bool)
    HOGCheck = np.zeros((nCams, 1), dtype=bool)

    #processing image of each camera
    for cc in range(1, nCams + 1, 1):
        hogImg = np.zeros((1,numHog)) #declare vector for saving hog features values of each strongest point
        diffDiameter = np.zeros((nC - crop * 2, 1))#declare vector for saving derivate of diamater values
        print("Cam: " + str(cc))
        print("Photo: " + str(1))
        img = openCV.imread("FaultCase/Cam"+str(cc)+"/"+str(1)+".png") #read image
        disk = openCV.getStructuringElement(openCV.MORPH_ELLIPSE, (2 * diamParameter - 1, 2 * diamParameter - 1))  # 15x15 matrix for closing matrix and obtaining a BW img            imgGray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY)  # rgb to gray
        imgGray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY) # convert image from bgr to gray scale
        c = openCV.Canny(imgGray, maxThrCanny, minThrCanny) #apply canny alghorithm on grayscaale image
        f = openCV.morphologyEx(c, openCV.MORPH_CLOSE, disk)  # f = BW img - close image --> img_as_ubyte convert scikit to openCV             
        blank_image, bH, bL = cropImg(f, img) #crop image
        
        diameter = np.subtract(bH, bL)#calculate diamater
        
        diff = np.gradient(diameter, axis=0)# calculate derivate
        
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
                diffDiameter[ii - crop] = np.mean(diff[0 : window, 0])
            elif ii >= (nC - window) :
                diffDiameter[ii - crop] = np.mean(diff[np.shape(diff)[0] - window : np.shape(diff)[0], 0])
            else:
                diffDiameter[ii - crop] = np.mean(diff[ii - window : ii + window])

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
                (H, hogImage) = feature.hog(roi, orientations=4, pixels_per_cell=(50, 50), cells_per_block=(2, 2), transform_sqrt=True, visualise=True, block_norm='L2-Hys', feature_vector=True)
                if len(H) == 16:
                    for j in range(0, np.shape(H)[0], 1):
                            vect[0,j] = H[j]
                    
                    hogImg = np.concatenate((hogImg, vect), axis=0)
                    
        hogImg = np.delete(hogImg, 0, 0)
        #check if image represented a fault rope or a healthy rope.
        if np.any(diffDiameter > diamLimit[cc - 1][0]) or np.any(diffDiameter < diamLimit[cc - 1][1]):
            diamCheck[cc - 1, 0] = True

        for i in range(0, np.shape(hogImg)[0], 1):
            for j in range(0, np.shape(hogImg)[1], 1):
                hogImg[i, j] = hogImg[i,j] - HOGMu[cc - 1, j]

        #apply same trasformation that were applicated in train function
        projectedHOG = np.dot(hogImg, HOGCoeff[cc - 1].T)
        m = np.mean(projectedHOG)

        if np.any(m > HOGLimit[cc - 1][0]) or np.any(m < HOGLimit[cc - 1][1]): 
            HOGCheck[cc - 1, 0] = True

        #print plot of mean of hog features and of derivate of diamater
        xAxis = np.arange(crop, nC - crop, 1)
        drawPlotDiameter(diffDiameter, diamLimit[cc - 1, 0], diamLimit[cc - 1, 1], xAxis, cc)
        
        x = np.arange(1, np.shape(projectedHOG)[0] + 1,1)
        drawPlotProjection(m, projectedHOG, HOGLimit[cc - 1, 0], HOGLimit[cc - 1, 1], x, cc)
    #print final table
    drawTable(diamCheck, HOGCheck)
    #return result
    return diamCheck, HOGCheck


def healthyCase(diamLimit, HOGLimit, HOGMu, HOGCoeff):
    print("HealthyCase:")
    nCams = 3 #number of photocameras
    diamParameter = 8 #diam_parameter for creating morphological element as disk with diameter of 15
    maxThrCanny = 150 #max threshold for appling canny algorithm on each image
    minThrCanny = maxThrCanny * 0.04 #min threshold for appling canny algorithm on each image
 
    #dimension of mobile window used to calculate derivate of diameter
    window = 150
    guard = 15
    crop = window + guard
    
    numStrongest = 20 #number of strongest points
    numHog = 16 #number of hog features values for each point

    nC = np.shape(openCV.imread("HealthyCase/Cam"+str(1)+"/"+str(1)+".png"))[1] #num columns of each image
    
    #declare this two vectors for print final result
    diamCheck = np.zeros((nCams, 1), dtype=bool)
    HOGCheck = np.zeros((nCams, 1), dtype=bool)

    #processing image of each camera
    for cc in range(1, nCams + 1, 1):
        hogImg = np.zeros((1,numHog)) #declare vector for saving hog features values of each strongest point
        diffDiameter = np.zeros((nC - crop * 2, 1))#declare vector for saving derivate of diamater values
        print("Cam: " + str(cc))
        print("Photo: " + str(1))
        img = openCV.imread("HealthyCase/Cam"+str(cc)+"/"+str(1)+".png") #read image
        disk = openCV.getStructuringElement(openCV.MORPH_ELLIPSE, (2 * diamParameter - 1, 2 * diamParameter - 1))  # 15x15 matrix for closing matrix and obtaining a BW img            imgGray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY)  # rgb to gray
        imgGray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY) # convert image from bgr to gray scale
        c = openCV.Canny(imgGray, maxThrCanny, minThrCanny) #apply canny alghorithm on grayscaale image
        f = openCV.morphologyEx(c, openCV.MORPH_CLOSE, disk)  # f = BW img - close image --> img_as_ubyte convert scikit to openCV             
        blank_image, bH, bL = cropImg(f, img) #crop image
        
        diameter = np.subtract(bH, bL)#calculate diamater
        
        diff = np.gradient(diameter, axis=0)# calculate derivate
        
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
                diffDiameter[ii - crop] = np.mean(diff[0 : window, 0])
            elif ii >= (nC - window) :
                diffDiameter[ii - crop] = np.mean(diff[np.shape(diff)[0] - window : np.shape(diff)[0], 0])
            else:
                diffDiameter[ii - crop] = np.mean(diff[ii - window : ii + window])

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
                (H, hogImage) = feature.hog(roi, orientations=4, pixels_per_cell=(50, 50), cells_per_block=(2, 2), transform_sqrt=True, visualise=True, block_norm='L2-Hys', feature_vector=True)
                if len(H) == 16:
                    for j in range(0, np.shape(H)[0], 1):
                            vect[0,j] = H[j]
                    
                    hogImg = np.concatenate((hogImg, vect), axis=0)
                    
        hogImg = np.delete(hogImg, 0, 0)
        #check if image represented a fault rope or a healthy rope.
        if np.any(diffDiameter > diamLimit[cc - 1][0]) or np.any(diffDiameter < diamLimit[cc - 1][1]):
            diamCheck[cc - 1, 0] = True

        for i in range(0, np.shape(hogImg)[0], 1):
            for j in range(0, np.shape(hogImg)[1], 1):
                hogImg[i, j] = hogImg[i,j] - HOGMu[cc - 1, j]

        #apply same trasformation that were applicated in train function
        projectedHOG = np.dot(hogImg, HOGCoeff[cc - 1].T)
        m = np.mean(projectedHOG)

        if np.any(m > HOGLimit[cc - 1][0]) or np.any(m < HOGLimit[cc - 1][1]): 
            HOGCheck[cc - 1, 0] = True

        #print plot of mean of hog features and of derivate of diamater
        xAxis = np.arange(crop, nC - crop, 1)
        drawPlotDiameterHealthy(diffDiameter, diamLimit[cc - 1, 0], diamLimit[cc - 1, 1], xAxis, cc)
        
        x = np.arange(1, np.shape(projectedHOG)[0] + 1,1)
        drawPlotProjectionHealthy(m, projectedHOG, HOGLimit[cc - 1, 0], HOGLimit[cc - 1, 1], x, cc)
    #print final table
    drawTable(diamCheck, HOGCheck)
    #return result
    return diamCheck, HOGCheck

  

            


      
    
    
    