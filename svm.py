
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import preprocessing



# download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
# info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# load data
bankdata = pd.read_csv("D:/bill_authentication.csv")  

# see the data
bankdata.shape  

# see head
bankdata.head()  

# data processing
X = bankdata.drop('Class', axis=1)  
y = bankdata['Class']  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

# train the SVM
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  

# predictions
y_pred = svclassifier.predict(X_test)  

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

# TODO output
"""

"""


# In[2]:



# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 
   
    # process
    import_iris.X = irisdata.drop('Class', axis=1)  
    import_iris.y = irisdata['Class']  
    
   

    
  
    
    #train
    from sklearn.model_selection import train_test_split  
    import_iris.X_train, import_iris.X_test, import_iris.y_train, import_iris.y_test = train_test_split(import_iris.X, import_iris.y, test_size = 0.20)  
    
        
    


# In[3]:


def color_map():
    colors = ('lightgreen' , 'blue' , 'red','yellow','pink')
    
    c_map = ListedColormap(colors[:len(np.unique(import_iris.y))])
    
    return c_map


# In[4]:


def polynomial_kernel():
    # TODO
    # NOTE: use 8-degree in the degree hyperparameter. 
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    polynomial_kernel.svcclf_poly = SVC(kernel='poly', degree=8)  
    polynomial_kernel.svcclf_poly.fit(import_iris.X_train, import_iris.y_train.ravel())

    #making preditions
    polynomial_kernel.y_pred = polynomial_kernel.svcclf_poly.predict(import_iris.X_test)  

    #evaluating performance
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(import_iris.y_test, polynomial_kernel.y_pred))  
    print(classification_report(import_iris.y_test, polynomial_kernel.y_pred))  
    


# In[5]:


def gaussian_kernel():
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    gaussian_kernel.svcclf_gaus= SVC(kernel='rbf')  
    gaussian_kernel.svcclf_gaus.fit(import_iris.X_train, import_iris.y_train)  
    #making preditions
    gaussian_kernel.y_pred = gaussian_kernel.svcclf_gaus.predict(import_iris.X_test) 
    
    #evaluating performance
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(import_iris.y_test,gaussian_kernel.y_pred))  
    print(classification_report(import_iris.y_test, gaussian_kernel.y_pred)) 


# In[6]:


def sigmoid_kernel():
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    sigmoid_kernel.svcclf_sig = SVC(kernel='sigmoid')  
    sigmoid_kernel.svcclf_sig.fit(import_iris.X_train, import_iris.y_train) 
    
    #making preditions
    sigmoid_kernel.y_pred = sigmoid_kernel.svcclf_sig.predict(import_iris.X_test)  
    
    #evaluating performance
    
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(import_iris.y_test, sigmoid_kernel.y_pred))  
    print(classification_report(import_iris.y_test, sigmoid_kernel.y_pred))  
    


# In[7]:




def test():
    #print(X.shape)
    import_iris()
 
    polynomial_kernel()
    gaussian_kernel()
    sigmoid_kernel()
    #plotting(import_iris.X_train,import_iris.y_train,polynomial_kernel.svcclf_poly ,"SVM")
    #plotting(X_train,y_train,gaussian_kernel.svcclf_gaus,"SVM")
    #plotting(X_train,y_train,sigmoid_kernel.svcclf_sig,"SVM")
    
    # NOTE: 3-point extra credit for plotting three kernel models.
   
    
test()


# In[8]:


# graphs (EXTRA CREDIT):
def mesh(X,y,r=0.08):
  
    X_min, X_max = X.min() - 1, X.max() + 1
    y_min, y_max = X.min() - 1, X.max() + 1

    XX , yy = np.meshgrid(np.arange(X_min,X_max,r),np.arange(y_min,y_max,r))
    return XX,yy

def contors(ax,classifier,XX,yy,**params):
    #Xpred = np.array([XX.ravel(),yy.ravel()] +[np.repeat(0, XX.ravel().size) for _ in range(2)]).T
    #c_map = color_map()
    
    result = classifier.predict(np.c_[XX.ravel(),yy.ravel()])
    result = result.reshape(XX.shape)
    #fig, ax = plt.subplots()
    
    out= ax.contourf(XX,yy,result, **params)
    return out
    
    
   


# In[9]:


from sklearn import svm, datasets
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, [2,3]]
y = iris.target

C= 1.0


# In[10]:




models = (SVC(kernel='sigmoid'),
          svm.SVC(kernel='rbf'),
          svm.SVC(kernel='poly'))

models = (clf.fit(X, y) for clf in models)

titles = ('SVC with Sigmoid kernel',
          'SVC with gaussian kernel(RBF))',
          'SVC with Polynomial kernel',)

fig, sub = plt.subplots(3,1,figsize=(15,20))


X0, X1 = X[:, 0], X[:, 1]
XX, yy = mesh(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    contors(ax, clf, XX, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

