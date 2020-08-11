# non- linear classification by using SVM / 비선형 분리 SVM
import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as metric
from sklearn.model_selection import cross_val_score,cross_validate
import pandas as pd

X,y = data.make_moons(n_samples=300,noise=0.16,random_state=42)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#dividing data
X_train, X_test, y_train, y_test = ms.train_test_split(X, y,test_size = 0.3, random_state = 100)

#SVM, kernel = linear
svm_clf = svm.SVC(kernel='linear',random_state=100)

#cross validation
scores= pd.DataFrame(cross_val_score(svm_clf,X,y,cv=5))
print(scores)
print('Average of cross validation by using linear kernel SVM:',scores.mean())

#SVM,kernel = 'rbf' => non-linear
rbf_svm_clf = svm.SVC(kernel='rbf')

#cross validation
scores2 = pd.DataFrame(cross_val_score(rbf_svm_clf,X,y,cv=5))
print(scores2)
print('Average of cross validation by using non-linear kernel SVM:',scores2.mean())

# Tuning parameters by using GridSearchCV
# Define the values to be tested as dictionary types
rbf_svm_clf = svm.SVC(kernel='rbf',random_state=100)

parameters={'C': [0.001,0.01,0.1,1,10,25,50,100],
            'gamma':[0.001,0.01,0.1,1,10,25,50,100]}
grid_svm=ms.GridSearchCV(rbf_svm_clf,param_grid=parameters,cv=5)
grid_svm.fit(X_train,y_train)

result=pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score',ascending=False)

print(result)
