import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNR
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CM

file=open("Summary.txt","a")



data=pd.read_csv("SuffledComplete.csv")



# you can tweak number of rows here instead of 15000000
rows=1500000
train=0.7
test=0.3



data=data.iloc[:rows,1:]



Model_Names=["Support Vector Classifier","Linear SVC",
             "GaussianNB","Random Forest Classifier","K Nearest Neighbours",
             "AdaBoost Classifier","Decision Tree Classifier"]


print(data.columns.values)



Models={1:SVC(C=1,kernel='rbf',gamma='scale',probability=True),
        2:LinearSVC(),
        3:GNB(),
        4:RFC(n_estimators=100),
        5:KNR(),
        6:ABC(base_estimator=RFC()),
        7:DTC()}


def RunModel(model,data,columns,Predict):
    X=data[columns]
    Y=data[Predict]


    X_train, X_test,y_train,y_test = train_test_split(X, Y,train_size=train,test_size=test,random_state=42)


    Model=model
    Model.fit(X_train, y_train)

    prediction = Model.predict(X_test)
    mse = (MSE(y_test, prediction))
    r2 = (R2(y_test, prediction))
    mae = (MAE(y_test, prediction))
    acc=AS(y_test,prediction)
    con_met=CM(y_test,prediction)
    return mse,r2,mae,acc,con_met




columns=['Length' ,'Time To Live', 'Protocol' ,'Source IP' ,'Destination IP',
 'SOURCE PORT' ,'Destination port']





for i in range(len(Model_Names)):
    mse, r2, mae, acc, cm = RunModel(Models[i + 1], data, columns, "Label")
    print("\nMODEL              -->            {5}\n\n"
          "Mean Squared Error is               {0}\n"
          "Coefficient Of Determination is     {1}\n"
          "Mean Absolute Error is              {2}\n"
          "Accuracy Score is                   {3}\n"
          "Confusin Matrix is              \n\n{4}\n"
          "\n\n\n\n\n\n".format(mse, r2, mae, acc, cm, Model_Names[i]), file=file)
