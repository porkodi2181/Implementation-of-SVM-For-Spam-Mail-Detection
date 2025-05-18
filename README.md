# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.

2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3. Split the dataset using train_test_split.

4. Calculate Y_Pred and accuracy.

5. Print all the outputs.

6. End the Program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PORKODI B
RegisterNumber: 212224240114
```
```
*/
import pandas as pd
data= pd.read_csv("C:/Users/admin/Desktop/INTR MACH/spam.csv", encoding= 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con= confusion_matrix(y_test, y_pred)
print(con)cl=classification_report(y_test,y_pred)
```

## Output:

## data_head():
![Screenshot 2025-05-18 123832](https://github.com/user-attachments/assets/fba45b8a-fbd6-4ed0-b9c4-82fd44659d46)

## data.isnull().sum:
![Screenshot 2025-05-18 124131](https://github.com/user-attachments/assets/a6fa0fe7-40ba-4486-bbd2-3280af15ff31)

## accuracy
![Screenshot 2025-05-18 124453](https://github.com/user-attachments/assets/07fb8687-c874-4ce4-ba0b-715931490d63)

## confusion matrix:
![Screenshot 2025-05-18 124609](https://github.com/user-attachments/assets/8247810d-c3e2-4bb1-b376-56f849065320)

## classification report

![Screenshot 2025-05-18 132345](https://github.com/user-attachments/assets/c721356c-1b6b-47d7-b985-adf4e42f8d6e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
