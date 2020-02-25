import sys
import pandas as pd

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 

buttonSelected=sys.argv[1]
pathTrain=sys.argv[2]
pathTest=sys.argv[3]
print(pathTest)
xSelected=sys.argv[4:-1]
ySelected=sys.argv[-1]
def TrainFunc():

  output_list=[]
  datasetTrain=pd.read_csv(pathTrain)
  xValues=datasetTrain[xSelected].values
  yValues=datasetTrain[ySelected].values
  XTrain,XTest,yTrain,yTest=train_test_split(xValues,yValues,test_size=.2,random_state=0)
  sc=StandardScaler()
  XTrain=sc.fit_transform(XTrain)
  XTest=sc.fit_transform(XTest)
  model=LogisticRegression()
  global fittedModel
  fittedModel=model.fit(XTrain,yTrain)
  predictions=fittedModel.predict(XTest)
  output_list.append("confusion_matrix::")
  output_list.append(confusion_matrix(yTest,predictions))
  output_list.append("Accuracy Score::")
  output_list.append(accuracy_score(yTest,predictions))
  
  return(output_list)
	
def TestFunc():
  output_list=[]
  datasetTest=pd.read_csv(pathTest)
  xValues=datasetTest[xSelected].values
  yValues=datasetTest[ySelected].values
  TrainFunc()
  predicted=fittedModel.predict(xValues)
  output_list.append("confusion_matrix::")
  output_list.append(confusion_matrix(yValues,predicted))
  output_list.append("Accuracy Score::")
  output_list.append(accuracy_score(yValues,predicted))
  return(output_list)
 
    
if buttonSelected=='Train':
	output=TrainFunc()
	print(output[0],"\n",output[1],"\n",output[2:])
else:
	print(TestFunc())	

		
'''

'''  	