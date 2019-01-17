""" 

“i pray thou, never fall  'i  love with me, for i am falser than vows made  'i  wine. ”  
yet  mine code is moe true than a person's last words.  Behold, " wine classification 
(and regression)"

"""

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# Preparing on the data(set)

# Step I - Importing the Dataset
from sklearn.datasets import load_wine
dataset=load_wine()

import pandas as pd
import numpy as np
x=pd.DataFrame(dataset.data,columns=[dataset.feature_names])
x['Class']=dataset.target
dataset=x

# Step II - Separating the Independent and Dependent variables
x=dataset.iloc[:,:13]
y=dataset.iloc[:,13].values.reshape(-1,1)


# Step III - Data Preprocessing
from sklearn.preprocessing import OneHotEncoder
oc=OneHotEncoder()
y=oc.fit_transform(y).toarray()

# Step IV - Train_Test_Split !
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Step V - Data Scaling 
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
sc2=StandardScaler()
x_train=sc1.fit_transform(x_train)
x_test=sc1.transform(x_test)
y_train=sc2.fit_transform(y_train)
y_test=sc2.transform(y_test)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Classification Begins:-
from keras.layers import Dense
from keras.models import Sequential

        
# Using the sequential model for our artificial neural network
cl=Sequential()
        
# Input Layer and the first hidden layer
cl.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_shape=(13,)))
        
# Second Hidden Layer
cl.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
        
# Third Hidden Layer
cl.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
        
# Output Layer
cl.add(Dense(units=3,kernel_initializer='uniform',activation='softmax'))
        
# Compiling the ANN
cl.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
# Fitting our model in the new ANN
cl.fit(x_train,y_train,epochs=200,batch_size=5)
        
        
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        

# Predicting the test values
y_pred=cl.predict(x_test)  
        
# Scaling the actual and predicted values again to convert them to their former originals
y_pred=sc2.inverse_transform(y_pred)
y_test=sc2.inverse_transform(y_test)
        
# 0.99999999 is almost equal class 1 and 0.00000001 is almost equal to 0
# The underneath instructions use the same concept to indicate the exact class names
        
# Converting to DataFrame for faster operation
y_pred=pd.DataFrame(y_pred)
y_pred[0]=y_pred[0].apply(lambda x: 1 if x>0.6 else 0)
y_pred[1]=y_pred[1].apply(lambda x: 1 if x>0.6 else 0)
y_pred[2]=y_pred[2].apply(lambda x: 1 if x>0.6 else 0)
y_test=pd.DataFrame(y_test)
        
# Checking for the error rate
from sklearn.metrics import mean_absolute_error
c_error=mean_absolute_error(y_pred,y_test)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# Regression 

# Step I - Preparing the data(set)

dat=load_wine()
dset=pd.DataFrame(dat.data,columns=[dat.feature_names])

# Step II - Separating the Independent and Dependent variables
b=dset['ash']
a=dset.drop(columns=['ash'],axis=1,level=0)
a=a.iloc[:,[0,2,3,5,6,7]]

# Step III - Train_Test_Split !
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2)

# Step IV - Data Scaling 
s1=StandardScaler()
s2=StandardScaler()
a_train=s1.fit_transform(a_train)
a_test=s1.transform(a_test)
b_train=s2.fit_transform(b_train)
b_test=s2.transform(b_test)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# Backward Elimination was used to determine which all input variables should be dumped


"""
import statsmodels.formula.api as sm

a=np.append(arr=np.ones((178,1)),values=a,axis=1)

x_opt=a[:,[1,3,4,6,7,8,12]]
rg=sm.OLS(endog=b,exog=x_opt).fit()
rg.summary()

"""

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# Regression Begins:-

        
# Using the sequential model for our artificial neural network  
reg=Sequential()
        
# The input layer and the first hidden layer
reg.add(Dense(units=4,input_shape=(6,)))
        
# The second hidden layer
reg.add((Dense(units=3)))
        
# The output layer
reg.add(Dense(units=1,))
        
# Compiling the ANN
reg.compile(optimizer='RMSprop',loss='logcosh')
           
# Fitting our model in the new ANN
reg.fit(a_train,b_train,epochs=100,batch_size=10)
        

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
        
# Prediction on test set
b_pred=reg.predict(a_test)
        
# Checking for the error rate
        
b_test=s2.inverse_transform(b_test)
b_pred=s2.inverse_transform(b_pred)
        
r_error=mean_absolute_error(b_pred,b_test) 


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        

print('\n\nThe mean arrant error for classification is ',c_error)
print('\n\nThe mean arrant error for regression is ',r_error)
print('Forsooth Commendable :D ')



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


""" Use %clear to clear the console for better visualization """

# THE MAIN SECTION
    
    
import time
import re

def i_want_to_be_amazed():
        
        
    print('\n\n\nRejoiceth, rejuvenate, revel, f\'r thou art about to receiveth the answ\'rs to thy wine. T hast nev\'r been so quite quaint but may receiveth like a toad, ugly and venemous. So, wend wisely')
    time.sleep(1)
    ch=input('Classify the stoup with\'i thy hand or hie deep to relapse :- ')
    time.sleep(1)
    
    while(re.search('[eE]xit',ch)==None):
        
        
        li=re.findall('[0-9]+[.]?[0-9]*',ch)
    
        if(re.search('[cC][lL][A-z]?[sS]?[sS][iI]?[fF][yY]',ch)!=None):         
            if(li!=13):
                li=[]
                print('Enter thy values and thou shall receive the quality')
                for g in range(0,13):
                    li.append(input('Enter value number %d - '%(g+1)))
            f=sc1.transform(np.array(li).reshape(1,-1))
            y_p=sc2.inverse_transform(((cl.predict(f))))
            if (y_p[0][0]>0.6):
                y_t='class 0'
            if y_p[0][1]>0.6:
                y_t='class 1'
            else:
                y_t='class 2'
                    
            print('Excited? the value comes to be ',y_t)
    
        
        if(re.search('[rR][A-z]?[lL][A-z]?[pP][sS][eE]?',ch)!=None):
            if(len(li)!=6):
               li=[]
               print('Enter thy values and thou shall receive the quality')
               for g in range(0,6):
                   li.append(input('Enter value number %d - '%(g+1)))
            e=s1.transform(np.array(li).reshape(1,-1))
            print('Excited? the value comes to be ',s2.inverse_transform(((reg.predict(e)))))
        
        ch=input(' Classify the stoup with\'i thy hand or hie deep to relapse....Again (Type exit to hie far from the code) ? ')   
      


# Type below For MENU










"""

The code comes to an end yet not our love to wot moe.  

~~ Made by Ram :D

"""
     












