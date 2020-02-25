import pandas as pd

import numpy as np

from sklearn.linear_model import ElasticNet

from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,mean_squared_error

from sklearn.model_selection import GridSearchCV,train_test_split

import sys



import math

buttonSelected=sys.argv[1]

pathTrain=sys.argv[2]

pathTest=sys.argv[3]



xSelected=sys.argv[4:-1]

ySelected=sys.argv[-1]

def Trainfunc():

 output_list=[]

 datasetTrain=pd.read_csv(pathTrain)

 xValues=datasetTrain[xSelected].values

 yValues=datasetTrain[ySelected].values



 XTrain,XTest,yTrain,yTest=train_test_split(xValues,yValues,test_size=0.2,random_state=0)

 param_grid={'alpha':np.random.uniform(0,100,10)}

 elastic=ElasticNet()

 global elasticModel

 elasticModel=GridSearchCV(elastic,param_grid,cv=10)

 elasticModel.fit(XTrain,yTrain)

 



 predicted=elasticModel.predict(XTest)

 

 

 elasticModelMse=mean_squared_error(yTest,predicted)

 output_list.append("MSE::")

 output_list.append(math.sqrt(elasticModelMse))

 output_list.append("R2 Score")

 output_list.append(r2_score(yTest,predicted))

 output_list.append("Best Parameter")

 output_list.append(elasticModel.best_params_)

 return(output_list)

 

 

def Testfunc():

 output_list=[]

 datasetTest=pd.read_csv(pathTest)

 xValues=datasetTest[xSelected].values

 yValues=datasetTest[ySelected].values

 Trainfunc()

 predicted=elasticModel.predict(xValues)

 output_list.append("Mean Square Error")

 output_list.append(mean_squared_error(yValues,predicted))

 output_list.append("R2 Score")

 output_list.append(r2_score(yValues,predicted))

 

 return(output_list)

 

if buttonSelected=='Train':

 print(Trainfunc())



else:

 print(Testfunc())