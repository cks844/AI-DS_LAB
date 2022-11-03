from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Loading data
irisData = load_iris()
# Create feature and target arrays
X = irisData.data
y = irisData.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
a=int(input("Enter the number for k value:")  )
knn = KNeighborsClassifier(n_neighbors=a)
print("Number datas for Preiction is:",len(X_test))
knn.fit(X_train, y_train)
print("Prediction Accuracy:",knn.score(X_test, y_test))
# Predict on dataset which model has not seen before
print("Predicted:",knn.predict(X_test))


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix\n",cm)


%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

if(cm[0,1] or cm[0,2] or cm[1,0] or cm[1,2] or cm[2,0] or cm[2,1]==0):
    print("The prediction is correct")
else:
    sum=cm[0,1]+cm[0,2]+cm[1,0]+cm[1,2]+cm[2,0]+cm[2,1]
    print("The prediction is wrong")
    print("Number of prediction is wrong=",sum)
