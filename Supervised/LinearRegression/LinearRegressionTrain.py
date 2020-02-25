import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 


buttonSelected=sys.argv[1]
pathTrain=sys.argv[2]
pathTest=sys.argv[3]
xColumns=sys.argv[4:-1]
yColumns=sys.argv[-1]
def TrainFunc():
	outputList=[]
	
	datasetTrain=pd.read_csv(pathTrain)
	datasetTest=pd.read_csv(pathTest)
	xValues=datasetTrain[xColumns].values
	yValues=datasetTrain[yColumns].values
	
	
	yValues=np.array(yValues).reshape(-1,1)
	xTrain,xTest,yTrain,yTest=train_test_split(xValues,yValues,test_size=0.2,random_state=0)
	model=LinearRegression()
	global fittedModel
	fittedModel=model.fit(xTrain,yTrain)
	
	predicted=fittedModel.predict(xTest)
	regressionModelMse=mean_squared_error(yTest,predicted)
	outputList.append("MSE::")
	outputList.append(math.sqrt(regressionModelMse))
	outputList.append("R Squared Value::")
	outputList.append(r2_score(yTest,predicted))
	outputList.append("Model coeffs are:")
	outputList.append(fittedModel.coef_[0])
	outputList.append(fittedModel.intercept_[0])
	
	return(outputList)
	
	
	
def TestFunc():
	outputList=[]
	pathTest=sys.argv[2]
	datasetTest=pd.read_csv(pathTest)
	xValues=datasetTest[xColumns].values	
	yValues=datasetTest[yColumns].values
	yValues=np.array(yValues).reshape(-1,1)
	TrainFunc()
	predicted=fittedModel.predict(xValues)
	regressionModelMse=mean_squared_error(yValues,predicted)
	outputList.append("MSE::")
	outputList.append(math.sqrt(regressionModelMse))
	outputList.append("R Squared Value::")
	outputList.append(r2_score(yValues,predicted))
	
	return(outputList)



if buttonSelected=='Train':
	output=TrainFunc()
	print("Result from Trainig dataset are")
	print(output[4:7],"\n",output[0:2],"\n",output[2:4],"\n")
	
	

else:
	output=TestFunc()
	print("Result from test dataset are")
	print(output[0:2],"\n",output[2:4])
		
	
	

'''	

plt.scatter(x,y,color='green')
	plt.plot(x,fittedModel.predict(x),color='black')
	plt.savefig('plot.png') 
	img = mpimg.imread('plot.png')
	x=plt.imshow(img)
	
	outputList.append(x)


	
	
	
	
	
'''	
