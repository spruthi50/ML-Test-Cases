import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split
import sys
buttonSelected=sys.argv[1]
pathTrain=sys.argv[2]
pathTest=sys.argv[3]
print(pathTest)
xSelected=sys.argv[4:-1]
ySelected=sys.argv[-1]

def Trainfunc():
 output_list=[]
 datasetTrain=pd.read_csv(pathTrain)
 xValues=datasetTrain[xSelected].values
 yValues=datasetTrain[ySelected].values
 XTrain,XTest,yTrain,yTest=train_test_split(xValues,yValues,test_size=0.2,random_state=0)
 param_grid={'n_neighbors':np.arange(1,50)}
 knn=KNeighborsClassifier()
 global knnCv
 knnCv=GridSearchCV(knn,param_grid,cv=10)
 knnCv.fit(XTrain,yTrain)
 yPred=knnCv.predict(XTest)
 output_list.append("Best Parameter")
 output_list.append(knnCv.best_params_)
 output_list.append("confusion matrix")
 output_list.append(confusion_matrix(yTest,yPred))
 output_list.append("Accuracy Score")
 output_list.append(accuracy_score(yTest,yPred))
 return(output_list)
 
def Testfunc():
 output_list=[]
 datasetTest=pd.read_csv(pathTest)
 xValues=datasetTest[xSelected].values
 yValues=datasetTest[ySelected].values
 Trainfunc()
 yPred=knnCv.predict(xValues)
 output_list.append("confusion Matrix")
 output_list.append(confusion_matrix(yValues,yPred))
 output_list.append("Accuracy Score")
 output_list.append(accuracy_score(yValues,yPred))
 return(output_list)
if buttonSelected=='Train':
	print(Trainfunc())
else:
	print(Testfunc())	 