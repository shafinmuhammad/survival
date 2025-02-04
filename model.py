import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df=pd.read_csv("passenger_survival_dataset.csv")
print(df.head())

df.info()

df.drop(["Name"],axis=1,inplace=True)
df.drop(["Passenger_ID"],axis=1,inplace=True)

df.info()
print(df.head())
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df.info()

print(df['Class'].unique())
print(df['Seat_Type'].unique())

df['Class'] = df['Class'].replace({'First': 1, 'Business': 2, 'Economy': 3})
df['Seat_Type'] = df['Seat_Type'].replace({'Window': 1, 'Middle': 2, 'Aisle': 3})



x=df.drop("Survival_Status",axis=1)
y=df["Survival_Status"]
 
print(x.head())
 
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

pickle.dump(model,open("model.pkl","wb"))
