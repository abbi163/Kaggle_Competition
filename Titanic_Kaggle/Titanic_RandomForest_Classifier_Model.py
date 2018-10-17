import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("E:/Pythoncode/Titanic_Kaggle/Data_Set/train.csv")
#print(df.describe())

# describe() is used for the following statistical data, {count, mean, std, min, 25%, 50%, 75% and max

df_survived = df.pop("Survived")

numeric_only = list(df.dtypes[df.dtypes != "object"].index)
# print(df[numeric_only])

df["Age"].fillna(df.Age.mean(), inplace = True)

# print(df[numeric_only])

model = RandomForestClassifier(n_estimators = 100)
model.fit(df[numeric_only], df_survived)
# print ("Train Accuracy ::", accuracy_score(df_survived, model.predict(df[numeric_only])))

df_test = pd.read_csv("E:/Pythoncode/Titanic_Kaggle/Data_Set/test.csv")

# print(df_test[numeric_only])

df_test["Age"].fillna(df_test.Age.mean(), inplace = True)


df_test = df_test[numeric_only].fillna(df_test.Age.mean()).copy()

df_survived_pred = model.predict(df_test[numeric_only])

submission = pd.DataFrame({

    "PassengerId": df_test["PassengerId"],
    "Survived": df_survived_pred
    })
submission.to_csv("E:/Pythoncode/Titanic_Kaggle/Data_Set/submission.csv", index = False)
print("end")
