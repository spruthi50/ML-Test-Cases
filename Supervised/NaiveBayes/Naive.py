import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split



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
	model=GaussianNB()
	global fittedModel
	fittedModel=model.fit(xTrain,yTrain)
	
	predicted=fittedModel.predict(xTest)
	outputList.append("Confusion Matrix is")
	outputList.append(confusion_matrix(yTest,predicted))
	outputList.append("Accuracy score is")
	outputList.append(accuracy_score(yTest,predicted))
	
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
	outputList.append("Confusion Matrix is")
	outputList.append(confusion_matrix(yValues,predicted))
	outputList.append("Accuracy score is")
	outputList.append(accuracy_score(yValues,predicted))
	
	return(outputList)



if buttonSelected=='Train':
	output=TrainFunc()
	print("Result from Trainig dataset are")
	print(output)
	
	

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
