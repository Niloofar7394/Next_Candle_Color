import pandas as pd
import numpy as np


data= pd.read_csv("USDJPY_Daily.csv")
s=data["target1"][1:]
s.reset_index(drop=True,inplace=True)
data.drop([len(data)-1],axis=0,inplace=True)
data["target1"]=s

data.drop(["date","target2"],axis=1,inplace=True)
data["target1"] = (data["target1"]>0)*1

ds = data.describe()

price_list=['open', 'high', 'low', 'close', 'open1', 'high1', 'low1', 'close1',
       'open2', 'high2', 'low2', 'close2', 'open3', 'high3', 'low3', 'close3',
       'open4', 'high4', 'low4', 'close4', 'open5', 'high5', 'low5', 'close5',
        'up5','mavg5','mavg4','dn4', 'up4', 'mavg2', 'up2', 'dn5',
        'dn3', 'mavg3', 'up3', 'dn0', 'mavg0', 'up0', 'dn2','up1']

new_data=pd.DataFrame()


#calculating logarithmic return of price data time series
for i in price_list:
    s=np.log(data[i][1:])
    s.reset_index(drop=True, inplace=True)
    m = np.log(data[i][:-1])
    m.reset_index(drop=True, inplace=True)
    new_data[i+"_LogDiff"] = s-m
    new_data[i]= data[i][1:]


other_features_list=['macd0', 'signal0', 'diff0', 'macd1', 'signal1', 'diff1', 'macd2',
       'signal2', 'diff2', 'macd3', 'signal3', 'diff3', 'macd4', 'signal4',
       'diff4', 'macd5', 'signal5', 'diff5', 'rsi0', 'rsi1', 'rsi2', 'rsi3',
       'rsi4', 'rsi5', 'pctB0', 'dn1', 'mavg1',
       'pctB1', 'pctB2', 'pctB3',
        'pctB4', 'pctB5']

for i in other_features_list:
    s=(data[i][1:]-ds[i]["mean"])/(ds[i]["std"])
    s.reset_index(drop=True, inplace=True)
    new_data[i+"Strd"]=s
    s = np.log(data[i][1:])
    s.reset_index(drop=True, inplace=True)
    m = np.log(data[i][:-1])
    m.reset_index(drop=True, inplace=True)
    new_data[i+"_LogDiff"]=s-m
    s=data[i][1:]
    s.reset_index(drop=True, inplace=True)
    new_data[i]=s

s=data["target1"][1:]
s.reset_index(drop=True, inplace=True)
new_data["target1"]=s

#drop unimportant features
new_data.drop(price_list,axis=1,inplace=True)
new_data.drop(other_features_list,axis=1,inplace=True)
new_data.drop(["macd2Strd","macd0Strd","signal0Strd","macd1Strd","signal1Strd","signal1_LogDiff",
               "diff1_LogDiff","up1_LogDiff","signal2_LogDiff","macd3_LogDiff","signal3Strd",
               "diff3Strd","macd4Strd","macd4_LogDiff"],axis=1,inplace=True)



s=new_data["high_LogDiff"][1:]
s.reset_index(drop=True,inplace=True)
new_data.drop([len(new_data)-1],axis=0,inplace=True)
new_data.reset_index(drop=True,inplace=True)
new_data["high_LogDiff"]=s

new_data.drop([0],axis=0,inplace=True)
new_data.reset_index(drop=True,inplace=True)


new_data=new_data.dropna()
new_data.reset_index(drop=True,inplace=True)
print("data len is :",len(new_data))
x_train,y_train = new_data.iloc[:int(len(new_data)*0.8),:-1],new_data.iloc[:int(len(new_data)*0.8),-1]
x_text, y_text = new_data.iloc[int(len(new_data)*0.8):,:-1],new_data.iloc[int(len(new_data)*0.8):,-1]

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

LR_classifier = LogisticRegression(random_state=0)
KNC_classifier = KNeighborsClassifier(n_neighbors=5)
DTC_classifier=DecisionTreeClassifier()
SVC_classifier = SVC(kernel="linear", random_state=0)
RFC_classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

LR_classifier.fit(x_train,y_train)
KNC_classifier.fit(x_train,y_train)
DTC_classifier.fit(x_train,y_train)
SVC_classifier.fit(x_train,y_train)
RFC_classifier.fit(x_train,y_train)

LR_p=LR_classifier.predict(x_text)
KNC_p=KNC_classifier.predict(x_text)
DTC_p=DTC_classifier.predict(x_text)
SVC_p=SVC_classifier.predict(x_text)
RFC_p=RFC_classifier.predict(x_text)




featureImportance = pd.DataFrame(DTC_classifier.feature_importances_,index=x_train.columns,columns=["FI"])
featureImportance.sort_values(["FI"],ascending=False,inplace=True)
print(featureImportance)


from sklearn.metrics import confusion_matrix


LR_cm= confusion_matrix(y_text,LR_p)
print("Logistic Regression Confusion Matrix:")
print(LR_cm)
print((LR_cm[0][0]+LR_cm[1][1])/(LR_cm[0][0]+LR_cm[1][1]+LR_cm[0][1]+LR_cm[1][0]))

KNC_cm= confusion_matrix(y_text,KNC_p)
print("KNeighbors Classifier Confusion Matrix:")
print(KNC_cm)
print((KNC_cm[0][0]+KNC_cm[1][1])/(KNC_cm[0][0]+KNC_cm[1][1]+KNC_cm[0][1]+KNC_cm[1][0]))

DTC_cm= confusion_matrix(y_text,DTC_p)
print("DecisionTree Classifier Confusion Matrix:")
print(DTC_cm)
print((DTC_cm[0][0]+DTC_cm[1][1])/(DTC_cm[0][0]+DTC_cm[1][1]+DTC_cm[0][1]+DTC_cm[1][0]))


SVC_cm= confusion_matrix(y_text,SVC_p)
print("SVC Classifier Confusion Matrix:")
print(SVC_cm)
print((SVC_cm[0][0]+SVC_cm[1][1])/(SVC_cm[0][0]+SVC_cm[1][1]+SVC_cm[0][1]+SVC_cm[1][0]))




RFC_cm= confusion_matrix(y_text,RFC_p)
print("RandomForest Classifier Confusion Matrix:")
print(RFC_cm)
print((RFC_cm[0][0]+RFC_cm[1][1])/(RFC_cm[0][0]+RFC_cm[1][1]+RFC_cm[0][1]+RFC_cm[1][0]))

