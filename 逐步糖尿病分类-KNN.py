# -*- encoding = utf-8 -*-

from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"]=["SimHei"]   # 用黑体显示中文
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号
diabetes_data=pd.read_csv("diabetes.csv",encoding="utf-8")
print(diabetes_data)

# print(diabetes_data.info(verbose=True))
# print(diabetes_data.describe().T)
diabetes_data_copy = diabetes_data.copy(deep = True)
#将0替换为nan
diabetes_data_copy[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = diabetes_data_copy[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)

# print(diabetes_data_copy.isnull().sum())
diabetes_data.hist(figsize=(20,20))
# plt.show()
#使用平均数和中位数填充
diabetes_data_copy["Glucose"].fillna(diabetes_data_copy["Glucose"].mean(), inplace = True)
diabetes_data_copy["BloodPressure"].fillna(diabetes_data_copy["BloodPressure"].mean(), inplace = True)
diabetes_data_copy["SkinThickness"].fillna(diabetes_data_copy["SkinThickness"].median(), inplace = True)
diabetes_data_copy["Insulin"].fillna(diabetes_data_copy["Insulin"].median(), inplace = True)
diabetes_data_copy["BMI"].fillna(diabetes_data_copy["BMI"].median(), inplace = True)
diabetes_data_copy.hist(figsize = (20,20))
plt.show()

print(diabetes_data.shape)
# print(diabetes_data.dtypes)
plt.figure(figsize=(5,5))
sns.set(font_scale=2)
sns.countplot("Glucose" ,data=diabetes_data)
plt.show()


import missingno as msno
msno.bar(diabetes_data)
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(diabetes_data,figsize=(25, 25))
plt.show()
sns.pairplot(diabetes_data_copy)
plt.show()
plt.figure(figsize=(12,10))
sns.heatmap(diabetes_data.corr(), annot=True,cmap ="RdYlGn")
plt.show()
plt.figure(figsize=(12,10))
sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ="RdYlGn")
plt.show()


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
       "BMI", "DiabetesPedigreeFunction", "Age"])
# print(X.head())
y = diabetes_data_copy.Outcome
# print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)



from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier

test_scores=[]
train_scores=[]
for i in range(1,15):
    knn=KNeighborsClassifier(i)
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


max_train_score=max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
# print("最大的训练数是 {} % 和 k = {}".format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
# print("最大的测试数是 {} % 和 k= {}".format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

plt.figure(figsize=(12,5))
sns.lineplot(range(1,15),train_scores,marker="*",label="训练分数")
sns.lineplot(range(1,15),test_scores,marker="o",label="测试分数")
plt.show()
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)


print(knn.score(X_test,y_test))
value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2,
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=X_test.values)

plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.title("KNN with Diabetes Data")
plt.show()

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=["True"], colnames=["Predicted"], margins=True)

y_pred = knn.predict(X_test)
print(y_pred)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt="g")
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr, label="Knn")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("Knn(n_neighbors=11) ROC curve")
plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred_proba))

from sklearn.model_selection import GridSearchCV

param_grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

