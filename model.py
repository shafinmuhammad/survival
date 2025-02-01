import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df=pd.read_csv("passenger_survival_dataset.csv")
df.head()

df.info()

df.drop(["Name","Passenger_ID"],axis=1,inplace=True)
df.info()

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df.info()

print(df['Class'].unique())
print(df['Seat_Type'].unique())
# one hot encoding in Class Economy as 1 
oe=OneHotEncoder()
df["Class"]=oe.fit_transform(df["Class"].values.reshape(-1,1)).toarray()
print(df['Class'])
print(df.info())
df.head()


x=df.drop("Survival_Status",axis=1)
y=df["Survival_Status"]

x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

pickle.dump(model,open("model.pkl","wb"))

print(df.head())