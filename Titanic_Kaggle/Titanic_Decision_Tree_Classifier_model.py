import pandas as pd

df = pd.read_csv('E:/Pythoncode/Titanic_Kaggle/Data_Set/train.csv')

# print (df.count())

# filling blanks space of Age column from mean of Age
df['Age'].fillna(df.Age.mean(),inplace = True)

#print (df.count())

X_train= df[['Pclass','Sex','Age','SibSp','Parch']]
y_train= df[['Survived']]

#print (X_train[0:5])

# importing test 

df_test = pd.read_csv('E:/Pythoncode/Titanic_Kaggle/Data_Set/test.csv')

# print (df.count())

df_test['Age'].fillna(df.Age.mean(),inplace = True)

#print (df.count())

X_test= df_test[['Pclass','Sex','Age','SibSp','Parch']]

#print(X_test['Age'].value_counts())


# importing decision tree classifier

from sklearn.tree import DecisionTreeClassifier

survivedTree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5)
survivedTree

survivedTree.fit(X_train,y_train)
predTree = survivedTree.predict(X_test)

submission = pd.DataFrame({

    "PassengerId": df_test["PassengerId"],
    "Survived": predTree
    })
submission.to_csv("E:/Pythoncode/Titanic_Kaggle/Data_Set/submission1.csv", index = False)
print("end")
