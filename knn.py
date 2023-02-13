import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Loading data
irisData = load_iris()

#features and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

n=int(input("Enter the number for k value:"))
knn = KNeighborsClassifier(n_neighbors=n)
print("Number of datas for Prediction is:",len(X_test))

#training
knn.fit(X_train, y_train)
print("Prediction Accuracy:",knn.score(X_test, y_test)*100)

# Prediction on test data
x=knn.predict(X_test)
plt.ylabel("Predicted Target")
plt.xlabel("Sl No")
slno=np.arange(0,len(x))
plt.bar(slno,x,color='g')
print("Actual Values   :",y_test)
print("Predicted Values:",x)
