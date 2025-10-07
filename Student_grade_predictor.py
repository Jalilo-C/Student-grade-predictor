####################### Student Grade predictor ( DATA SCIENCE PROJECT ) #################

import pandas as pd
from sklearn.model_selection import train_test_split # split into training and testing
from sklearn.linear_model import LinearRegression # ML model
from sklearn.metrics import mean_squared_error # give how good your model is
from sklearn.metrics import r2_score # tells how well the model explains the variance


## Load the data
data=pd.read_csv('student-mat.csv', sep=';')
features=data[["studytime", "absences", "G1", "G2"]] # give the features named here of every student
final_grades=data[["G3"]] # give their final grades

## Split into training/test
features_train,features_test,final_grades_train,final_grades_test=train_test_split(features,final_grades,test_size=0.2) # 80% training 20% test

## Train the model 
model=LinearRegression() # creation
model.fit(features_train,final_grades_train)

## Make a prediction
final_grades_predict=model.predict(features_test)

## Evaluate with R² score and Mean Squared Error
squared_error=mean_squared_error(final_grades_test,final_grades_predict)
r2=r2_score(final_grades_test,final_grades_predict) # closer to 1 the better
print(f"Mean squared error: {squared_error} and R2 score: {r2}")
print()

## Predict a student's grade from sample input
print("Let's now predict the final grade of a student based on the features:")

print("Studytime:")
print("1. < 2 hours per week")
print("2. 2 to 5 hours per week")
print("3. 5 to 10 hours per week")
print("4. > 10 hours per week")
studytime=int(input("How much do you study per week ( eg: 1 for less than 2 hours ): "))
while studytime not in [1,2,3,4]:
    print("Oups, wrong answer")
    studytime=int(input("How much do you study per week ( eg: 1 for less than 2 hours ): "))
print()

absences=int(input('How much days of school did you miss ( less than 75 ): '))
while absences not in range(0,76):
    print("Oups, wrong answer")
    absences=int(input('How much days of school did you miss ( less than 75 ): '))
print()

G1=int(input("how much did you got in the first-period grade ( between 0 and 20 )? "))
while G1 not in range(0,21):
    print("Oups, wrong answer")
    G1=int(input("how much did you got in the first-period grade ( between 0 and 20 )? "))
print()

G2=int(input("how much did you got in the second-period grade ( between 0 and 20 )? "))
while G2 not in range(0,21):
    print("Oups, wrong answer")
    G2=int(input("how much did you got in the second-period grade ( between 0 and 20 )? "))
print()

sample=[[studytime,absences,G1,G2]]
G3_predict=model.predict(sample)
print("Predicted grade:",round(G3_predict[0][0],2))
