#ECG Classifier
#By Nathan Andrew Deguara
#S16109197
#Final Year Honours Project

#Import required librarys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import SymbolicFourierApproximation
import warnings
import time

#Removes warning messages from output window
warnings.filterwarnings("ignore")

#Start the runtime timer
start_time = time.clock()

#Read in the data
train = pd.read_csv('Datasets/ECG5000/ECG5000_TRAIN.csv')
test = pd.read_csv('Datasets/ECG5000/ECG5000_TEST.csv')

#Assign to x and y train/test
Xtr = train.iloc[:,1:]
ytr = train.iloc[:,:1]
Xte = test.iloc[:,1:]
yte = test.iloc[:,:1]

#Class for data transformation
class Data_Transformer():
    SAX = SymbolicAggregateApproximation(strategy= 'uniform', alphabet= 'ordinal')
    Xtr_SAX = SAX.fit_transform(Xtr)
    Xte_SAX = SAX.fit_transform(Xte)
    
    SFA = SymbolicFourierApproximation(alphabet= 'ordinal')
    Xtr_SFA = SFA.fit_transform(Xtr)
    Xte_SFA = SFA.fit_transform(Xte)

#Class to define the models and their optimal parameters
class Models():
    RF_parameters = {'bootstrap': [True], 'max_depth': [10], 'min_samples_leaf':[1], 'min_samples_split': [6], 'n_estimators': [100,200,300], 'random_state': [10]}
    RF_model = GridSearchCV(RandomForestClassifier(), RF_parameters, n_jobs = -1, cv=2)
    RF_model.fit(Data_Transformer.Xtr_SAX, ytr.values.ravel())
    
    SVC_parameters = {'C':[1,3,5],'kernel': ['rbf','linear'], 'gamma': ['scale'],'random_state': [10], 'probability': [True]}
    SVC_model = GridSearchCV(SVC(), SVC_parameters, n_jobs = -1, cv=2)
    SVC_model.fit(Data_Transformer.Xtr_SAX, ytr.values.ravel())
    
    LR_model = LogisticRegression(random_state=10)
    LR_model.fit(Data_Transformer.Xtr_SAX, ytr.values.ravel())
    
#Class for ensemble voting
class Ensemble():
    #The models to use
    estimators = [("RF", Models.RF_model), ("SVC", Models.SVC_model), ("LR", Models.LR_model)]
    #The weights to assign to each model
    weights=[2,2,1]
    
    vote = VotingClassifier(estimators = estimators, voting = "soft", weights = weights)
    vote.fit(Data_Transformer.Xtr_SAX, ytr.values.ravel())
    
#Main/run the code
scores = cross_val_score(Models.RF_model, Data_Transformer.Xte_SAX, yte, scoring='accuracy', cv=5)
print("RF Accuracy: %0.2f%%" %(scores.mean()*100))
scores = cross_val_score(Models.SVC_model, Data_Transformer.Xte_SAX, yte, scoring='accuracy', cv=5)
print("SVC Accuracy: %0.2f%%" %(scores.mean()*100))
scores = cross_val_score(Models.LR_model, Data_Transformer.Xte_SAX, yte, scoring='accuracy', cv=5)
print("LR Accuracy: %0.2f%%" %(scores.mean()*100))
scores = cross_val_score(Ensemble.vote, Data_Transformer.Xte_SAX, yte, scoring='accuracy', cv=5)
print("Enemble Accuracy: %0.2f%%" %(scores.mean()*100))

#Display the runtime
print("Runtime: %.2f seconds" %(time.clock()-start_time))