
# coding: utf-8

# In[3]:

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn import metrics
from sklearn import ensemble
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[29]:

import pickle


# In[37]:

rng=np.random.RandomState(21342)


# In[30]:

path=r'/Users/XW/Desktop'


# In[31]:

X=pickle.load(open(path+'/x.pkl','r'))


# In[39]:

X = np.array(map(lambda x: np.concatenate((np.outer(x,x).reshape(len(x)**2),x)) ,X))


# In[42]:

Y=pickle.load(open(path+'/y.pkl','r'))


# In[44]:

X=preprocessing.scale(X)


# In[47]:

n_total=len(Y)
index=range(n_total)
rng.shuffle(index)
xtrain=X[index[0:n_total/3*2]]
ytrain=Y[index[0:n_total/3*2]]
xtest=X[index[n_total/3*2::]]
ytest=Y[index[n_total/3*2::]]
ratio=float(len([i for i in Y if i!='Yes']))/n_total


# In[51]:

from sklearn import svm,grid_search
from sklearn import preprocessing
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[68]:

def eval(ytest, ypred):
    print 'Confusion Matrix:'
    print confusion_matrix(ytest, ypred)
    print 'Accuracy: %f' % ( metrics.accuracy_score(ytest, ypred))
    print metrics.classification_report(ytest, ypred)


# In[54]:

#Logistic Regression
Logistic_model = LogisticRegression(penalty='l1', C=0.1, class_weight='balanced')
Logistic_model.fit(xtrain, ytrain)


# In[55]:

ypred = Logistic_model.predict(xtest)


# In[69]:

eval(ytest, ypred)


# In[70]:

#naive bayes
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)
NBpred = gnb.predict(xtest)
eval(ytest, NBpred)


# In[75]:

#Adaboost 
def AdaBoost(xtrain, xtest, ytrain, ytest):
    depth=75
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=depth)
    model.fit(xtrain, ytrain)
    print 'Adaboost with depth %d' %depth 
    print 'Test Performance'
    eval(ytest, model.predict(xtest))
    print 'Train Performance'
    eval(ytrain, model.predict(xtrain))


# In[76]:

AdaBoost(xtrain, xtest, ytrain, ytest)


# In[83]:

#randomforest
def random_forest(xtrain, xtest, ytrain, ytest, cv=False):
    if cv==False:
        k=400
        d=9
        rf=RandomForestClassifier(n_estimators=k,max_depth=d,criterion='gini',class_weight='balanced',max_features=10)
        rf.fit(xtrain,ytrain)
        pred=rf.predict(xtest)
        print 'Random Forest with %d tree, %d depth' %(k,d)
        print 'Test Performance'
        eval(ytest, pred)
        print 'Train Performace'
        eval(ytrain, rf.predict(xtrain))
    else:
        para={'n_estimators':[200,400],"max_depth":[3,4,5,6,7,9],'class_weight':['balanced',None],'max_features':['auto','log2',10]}
        rf=RandomForestClassifier()
        clf=grid_search.GridSearchCV(rf,para,scoring=make_scorer(metrics.accuracy_score,greater_is_better=True))
        clf.fit(xtrain,ytrain)
        print clf.best_params_
        print clf.best_score_


# In[84]:

random_forest(xtrain, xtest, ytrain, ytest, cv=True)


# In[113]:

clf=RandomForestClassifier(n_estimators=200,max_depth=4,class_weight='balanced')
clf=clf.fit(xtrain, ytrain)
feature_importance=clf.feature_importances_


# In[115]:

model=SelectFromModel(clf,prefit=True)


# In[117]:

model.transform(xtrain)


# In[111]:

#SVM
def SVM(xtrain, xtest, ytrain, ytest, cv=False):
    if cv==True:
        param={'kernel':['rbf'], 'C':[0.1,10,1000,10000,100000],'gamma':[0.01, 0.1, 1, 10, 100, 1000, 5000]}
        svr=svm.SVC()
        clf=grid_search.GridSearchCV(svr,param,scoring=make_scorer(metrics.accuracy_score,greater_is_better=True))
        clf.fit(xtrain, ytrain)
        print clf.best_score_
        print clf.best_params_
    else:
        model_svm=svm.SVC(C=10,kernel='rbf',gamma=1000)
        model_svm.fit(xtrain,ytrain)
        print 'Test Performance' 
        eval(ytest, model_svm.predict(xtest))
        print 'Train Performance'
        eval(ytrain, model_svm.predict(xtrain))


# In[112]:

SVM(xtrain, xtest, ytrain, ytest,cv=True)


# In[ ]:



