# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
data.head:
<img width="1035" height="182" alt="image" src="https://github.com/user-attachments/assets/1494d269-b150-47d7-91f0-25d460184177" />
data1.head:
<img width="1031" height="192" alt="image" src="https://github.com/user-attachments/assets/81331b2d-4394-4be5-b1a3-1277ccf09bc9" />
























data1.isnull().sum():






<img width="321" height="376" alt="image" src="https://github.com/user-attachments/assets/22ea55f8-f885-4fc2-96d6-70368c444e37" />























data1.duplicated().sum():









<img width="68" height="37" alt="image" src="https://github.com/user-attachments/assets/9b8feae2-3c18-422f-b40f-762d34a3e690" />











data1:




<img width="1030" height="205" alt="image" src="https://github.com/user-attachments/assets/3903450a-b0ab-41bf-bac3-e5c98b7e1748" />














x:








<img width="1033" height="227" alt="image" src="https://github.com/user-attachments/assets/75404925-11a2-4c4f-b92d-89ef27085e36" />














y:



























<img width="526" height="316" alt="image" src="https://github.com/user-attachments/assets/6c23e54b-fbb6-44eb-9628-44e9055f44d9" />








y_pred:


















<img width="901" height="56" alt="image" src="https://github.com/user-attachments/assets/e05284e6-9bfd-4108-9133-dceaee2843c8" />
















Accuracy:
















<img width="367" height="36" alt="image" src="https://github.com/user-attachments/assets/b2576648-3d82-4918-8e97-7b3584440c29" />

























Confusion:























<img width="146" height="81" alt="image" src="https://github.com/user-attachments/assets/8633a449-a560-42f7-b50f-af7e9fc99ef2" />
















Classification Report1:






















<img width="720" height="237" alt="image" src="https://github.com/user-attachments/assets/f8fbdc63-fdf1-483e-a89f-4624f71969b5" />







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
