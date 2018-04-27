#import train and check function
from RFD_Train import train
from RFD_Check import check, healthyCase

#call train and check function to find out if rope is healthy or fault
diamLimit, HOGMu, HOGCoeff, HOGLimit = train()
diamCheck, HOGCheck = check(diamLimit, HOGLimit,HOGMu, HOGCoeff)
diamCheckHealthy, HOGCheckHealthy = healthyCase(diamLimit, HOGLimit,HOGMu, HOGCoeff)
#print result on output console
print(diamCheck)
print("---")
print(HOGCheck)
print("****")
print(diamCheckHealthy)
print("---")
print(HOGCheckHealthy)

