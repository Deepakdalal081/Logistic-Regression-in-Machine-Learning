# 1. Define a Classification Task

# Next day is going to be a up or down 

#import Libeary 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt 
import seaborn as sns 
import warnings 
import yfinance as yf 

warnings.filterwarnings("ignore")

# 2. Read the dataset 

 
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days = 1000)

stock_data = yf.download("AXISBANK.NS", start= start_date, end= end_date, interval = "1D")


#print(stock_data)


# 3. Generate Target Value 

stock_data["Return"] = np.log(stock_data["Close"]/stock_data["Close"].shift(1))

#If next Close is up than yesterday make it 1 otherwise -1 

stock_data["Target"] = np.where(stock_data["Return"].shift(-1)> 0, 1, -1)
#print(stock_data["Target"].value_counts())


# 4. Feature Selection 
#Close price play a imporant role in next day movement 
plt.fig = figsize = (10,5)
#sns.scatterplot(x = stock_data["Return"], y = stock_data["Volume"])

# 5. Feature Extraction 
# Close price is not enough we need more feature than we will extract some from the exsiting ones
 
#Storing the Values
feature_list = []

#Rolling Standard Deviation
for i in range(5,20,5):
    col_name = "SD" + str(i)
    stock_data[col_name] = stock_data["Close"].rolling(window = i).std()
    feature_list.append(col_name)
       
#Rolling Moving Average 
for i in range(1, 20 , 3):
    col_name = "MA" + str(i)
    stock_data[col_name] = stock_data["Close"].rolling(window = i).mean()
    feature_list.append(col_name)
    
#Rolling pct change
for i in range(1,21,4):
   col_name = "" + str(i)
   stock_data[col_name] = stock_data["Close"].pct_change().rolling(window = i).sum()
   feature_list.append(col_name)    

# Rolling Moving average of Volume    
  
col_name = "4RV"
stock_data[col_name] = stock_data["Volume"].rolling(4).mean()
feature_list.append(col_name)

#- Difference between close and open
col_name = "CO"
stock_data[col_name] = stock_data["Close"] - stock_data["Open"]
feature_list.append(col_name)

#Removing the NaN Values 
stock_data.dropna(inplace = True)

#Shifting the Target column at the end 
stock_data = stock_data[feature_list + ["Target"]]


#6. Generate Train-Test Datasets
#Out of all the data we will take 70 % data for calculation and based on that we will find the value of test data 

from sklearn.model_selection import train_test_split
X = stock_data[feature_list].iloc[:-1] # X_train  data 
y = stock_data.iloc[:-1]["Target"] # y_test data 
X_train, X_test , y_train, y_test = train_test_split(X,y,train_size= .70 ,shuffle=False)



#print(X_train.shape, X_test.shape)    


# 7. Feature Scaling
# ML require data in the normalized form 

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train) # normalizing the data 
X_test_scaled = scalar.transform(X_test) # in this using the train data just for the nornalizing not using for the testing 

x_scalled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns)

#Plotting the noramlized data to check weather the x & y axis have scale nearer or different  
#sns.pairplot(x_scalled_df[["SD5", "MA7", "4RV"]])



#8. Build Model

from sklearn.linear_model import LogisticRegression

# Every time i run the code i will get the same result not the different 
model = LogisticRegression(random_state= 1) # random_state = 1 


# 9. Train the Model 

# 70 % data taken for the training purpose use those values  
model.fit(X_train_scaled, y_train)


# 10 . Predict the value 

# Predting the y train data based in the X_train_scaled data 
y_pred_scalled = model.predict(X_train_scaled)

# how much is the accuracy in the y_train data compared with the true result 
print(f"\nThe model accuracy for trainig data ", model.score(X_train_scaled, y_train))

# Predicting the y Test data 
y_pred = model.predict(X_test_scaled)

# How much is the accuracy in the Y_test data based on the X scalled data  
print("The model accuracy for testing data ", model.score(X_test_scaled, y_test))











