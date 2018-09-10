from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


iris = datasets.load_iris() 
X,y = iris.data,iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)



##########################################################################3
neighbors = np.arange(1, 26)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
###############################################################################################

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.25, random_state=6, stratify=y)

neighbors = np.arange(1, 26)
test_accuracy2 = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train1,y_train1)
    #Compute accuracy on the training set
    test_accuracy2[i] = knn.score(X_test1,y_test1)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy2, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#################################################################################################

max_depth_range = range(1,11)
scores=[]
for a in max_depth_range:
    dt = DecisionTreeClassifier(criterion='gini', max_depth=a, random_state=6)
    dt.fit(X_train,y_train)
    y_pred= dt.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    print (accuracy_score(y_test, y_pred))
plt.title('TreeClassifier: accuracy according to max_depth')
plt.plot(max_depth_range, scores, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()    
    
print("My name is Xiaoyu Yuan")
print("My NetID is: 664377413")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
