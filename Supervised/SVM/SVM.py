import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

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
 model=SVC()
 np.random.seed(0)
 param_grid = {'C': np.random.uniform(0,100,15),  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
 global svmModel
 svmModel=GridSearchCV(model,param_grid,cv=10)
 svmModel.fit(XTrain,yTrain)
 prediction=svmModel.predict(XTest)
 output_list.append("confusion_matrix")
 output_list.append(confusion_matrix(yTest,prediction))
 output_list.append("Accuracy Score::")
 output_list.append(accuracy_score(yTest,prediction))
 output_list.append("Classification Report")
 output_list.append(classification_report(yTest,prediction))
 output_list.append("Best parameter")
 output_list.append(svmModel.best_params_)
 output_list.append("Best estimator")
 output_list.append(svmModel.best_estimator_)
 return(output_list)
 
def Testfunc():
 output_list=[]
 datasetTest=pd.read_csv(pathTest)
 xValues=datasetTest[xSelected].values
 yValues=datasetTest[ySelected].values
 Trainfunc()
 prediction=svmModel.predict(xValues)
 output_list.append("Accuracy Score::")
 output_list.append(accuracy_score(yValues,prediction))
 
 output_list.append("Classification Report")
 output_list.append(classification_report(yValues,prediction))
 
 return(output_list)
 
if buttonSelected=='Train':
 print(Trainfunc())
 
else:
 print(Testfunc()) 