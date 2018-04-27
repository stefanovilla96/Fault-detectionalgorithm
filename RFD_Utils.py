import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cropImg(f, img):
    nC = np.shape(f)[1] #number of columuns of binary image
    bL = np.zeros((nC, 1), dtype=int) #declare vector where values of row about upper border are saved
    bH = np.zeros((nC, 1), dtype=int) #declare vector where values of row about lower border are saved
    borderRC = np.nonzero(f) #positions of 1 in f[row][col]
    # borderRC[0]->rows ; borderRC[1]->columns
    ini = np.amin(borderRC[0])#lowest border
    fin = np.amax(borderRC[0])#uppest border
    #create blank image for cropping operation
    blank_image = np.zeros((fin - ini, nC, 3), dtype=np.uint8)  # create black black image
    blank_image[:, :, :] = (255, 255, 255)  # transform black blanck image to white image
    #copy the rope in white image
    for i in range (0, nC, 1):
        bL[i] = np.amin(borderRC[0][np.argwhere(borderRC[1] == i)])
        bH[i] = np.amax(borderRC[0][np.argwhere(borderRC[1] == i)])
        blank_image[int(bL[i]) - ini: int(bH[i]) - ini, i,:] = img[int(bL[i]) : int(bH[i]), i,:]
    
    return blank_image, bH, bL

def drawPlotDiameter(diffDiameter, upperLimit, LowerLimit, xAxis, camera):

    plt.figure(1)
    plt.title("Diameter derivative threshold from camera " + str(camera))#title of plot
    plt.xlabel("Rope length [Px]")# label of x axis
    plt.ylabel("Rope diameter derivative [Px/Px]") #label of y axis
    plt.plot(xAxis, diffDiameter, xAxis,upperLimit * np.ones((np.shape(xAxis)[0],1)), 'r--', xAxis,LowerLimit * np.ones((np.shape(xAxis)[0],1)),'r--')#plot diameter derivate values and diameter derivate limit values
    plt.draw()
    plt.show()

def drawPlotProjection(m, projectedHOG, upperLimit, lowerLimit, xAxis, camera):

    plt.figure(1)
    plt.title("HOG features mean from camera " + str(camera)) #title of plot
    plt.ylabel("Projected features")# label of y axis
    plt.plot(xAxis, m * np.ones((np.shape(xAxis)[0],1)) , xAxis,upperLimit * np.ones((np.shape(xAxis)[0],1)), 'r--', xAxis,lowerLimit * np.ones((np.shape(xAxis)[0],1)),'r--')#plot mean of hog features values and hog feautures limit values
    ymin, ymax = plt.ylim()  # return the current ylim
    plt.yticks(np.arange(np.around(ymin, decimals=2), np.around(ymax, decimals=2), step=0.02))
    plt.draw()
    plt.show()

def drawTable(diamCheck, HOGCheck):
    
    rows = ("Camera 1", "Camera 2", "Camera 3") #label of rows
    columns = ('Diameter', 'HOG')#label of colums
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    result = np.zeros((3,2),dtype=bool)
    
    for i in range(0, np.shape(result)[0], 1):
        result[i,0] = diamCheck[i,0]
    for i in range(0, np.shape(result)[0], 1):
        result[i,1] = HOGCheck[i,0]
    
    df = pd.DataFrame(result, columns=list('HD'))
    ax.table(cellText=df.values, rowLabels=rows, colLabels=columns, loc='center')
    fig.tight_layout()
    
    #print comments under the table
    for i in range(0, np.shape(diamCheck)[0], 1):
        if(diamCheck[i]):
            plt.text(-0.02, -0.02, "Photo from camera %d represented a fault rope for derivate of diameter values" %(i + 1))
        else:
            if(HOGCheck[i]):
                plt.text(-0.02, -0.025, "Photo from camera %d represented a fault rope for HOG Features values"%(i + 1))
            else:
                plt.text(-0.02, -0.02 - 0.005 * i, "Photo from camera %d represented a healthy rope both for HOG Features values and derivate of diameter values"%(i + 1))
    
    plt.show()

def drawPlotDiameterHealthy(diffDiameter, upperLimit, LowerLimit, xAxis, camera):

    plt.figure(1)
    plt.title("Healthy Case - Diameter derivative threshold from camera " + str(camera))#title of plot
    plt.xlabel("Rope length [Px]")#label of x axis
    plt.ylabel("Rope diameter derivative [Px/Px]") #label of y axis
    plt.plot(xAxis, diffDiameter, xAxis,upperLimit * np.ones((np.shape(xAxis)[0],1)), 'r--', xAxis,LowerLimit * np.ones((np.shape(xAxis)[0],1)),'r--')#plot diameter derivate values and diameter derivate limit values
    plt.draw()#draw and show plot
    plt.show()

def drawPlotProjectionHealthy(m, projectedHOG, upperLimit, lowerLimit, xAxis, camera):

    plt.figure(1)
    plt.title("Healthy Case - HOG features mean from camera " + str(camera))#title of plot
    plt.ylabel("Projected features")#label of y axis
    plt.plot(xAxis, m * np.ones((np.shape(xAxis)[0],1)) , xAxis,upperLimit * np.ones((np.shape(xAxis)[0],1)), 'r--', xAxis,lowerLimit * np.ones((np.shape(xAxis)[0],1)),'r--')#plot mean of hog features values and hog feautures limit values
    ymin, ymax = plt.ylim()  # return the current ylim
    #plt.ylim((ymin * 2, ymax * 2))   # set the ylim to ymin, ymax
    plt.yticks(np.arange(np.around(ymin, decimals=2), np.around(ymax, decimals=2), step=0.02))
    plt.draw()#draw and show plot
    plt.show()


