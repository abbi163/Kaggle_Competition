import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('E:/Sample data/Kaggle/Titanic/train.csv', index_col = 0)

# size of the figure which pops out once the plot is shown.
fig = plt.figure(figsize=(18,10))

# by using subplot2grid method we can plot multiple graphs in one sheet. 
plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind='bar', alpha = 0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind='bar', alpha = 0.5)
plt.title("Class")


plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha = 0.1)
plt.title("Survived with respect to Age")



plt.subplot2grid((2,3),(1,0), colspan = 2)
for x in [1,2,3]:
	df.Age[df.Pclass == x].plot(kind ="kde")
plt.title("Class wrt Age")
plt.legend(("1st", "2nd", "3rd"))
# plt.legend(("1st", "2nd", "3rd")) implies on graph it will be shown which one is 1st ,2nd and 3rd
plt.show()

print(df.count())
# df.count() method is used to count total number of data in each of the columns
