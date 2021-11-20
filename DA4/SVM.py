# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Reading Dataset
x = [1, 2, 1, 1, 4, 5, 5, 6]
y = [1, 1, -1, 2, 0, 1, -1, 0]

plt.scatter(x,y)
plt.show()

X = np.array([[1,1],
             [2,1],
             [1,-1],
             [1,2],
             [4,0],
             [5,1],
             [5,-1],
             [6,0]])

y = [0,1,0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X,y)

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label = "non weighted div")
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.title("Linear SVM classifier")
plt.legend()
plt.show()