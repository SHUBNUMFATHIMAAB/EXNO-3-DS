## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd 

df= pd.read_csv("/content/Encoding Data.csv")

df

<img width="477" height="464" alt="image" src="https://github.com/user-attachments/assets/9517a253-ad70-4492-afa0-78179f22c1d9" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm= ['Hot','Warm','Cold']

e1= OrdinalEncoder (categories=[pm])

e1.fit_transform (df[["ord_2"]])

<img width="231" height="246" alt="image" src="https://github.com/user-attachments/assets/c39bd867-dbf9-4a07-8e6e-c0a896c21865" />

df['bo2']= e1.fit_transform(df[["ord_2"]])

df

<img width="484" height="468" alt="image" src="https://github.com/user-attachments/assets/8abd2c23-dc88-487f-a1cb-7a1719342ab3" />

le= LabelEncoder()

dfc= df.copy()

dfc['ord_2']=le.fit_transform (dfc['ord_2'])

dfc

<img width="479" height="444" alt="image" src="https://github.com/user-attachments/assets/0f7a7fbf-5b5e-40bf-af6c-17f6ecc7c7dc" />

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2= pd.concat([df2,enc],axis=1)

df2

<img width="643" height="454" alt="image" src="https://github.com/user-attachments/assets/767dedbc-9902-4a3f-95bb-6e1ad320378d" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="903" height="469" alt="image" src="https://github.com/user-attachments/assets/00a8005a-501d-496b-b377-4426773e0e2d" />

from category_encoders import BinaryEncoder

df= pd.read_csv("/content/data.csv")

df

<img width="679" height="464" alt="image" src="https://github.com/user-attachments/assets/77ce0430-e546-4939-b74c-2b23db9fc624" />

be= BinaryEncoder()

nd= be.fit_transform(df['Ord_2'])

df

<img width="688" height="512" alt="image" src="https://github.com/user-attachments/assets/f93b7ee5-bd05-46e8-87aa-a3618576a62f" />

dfb= pd.concat([df,nd],axis=1)

dfb

<img width="959" height="469" alt="image" src="https://github.com/user-attachments/assets/d891107b-d24f-4c79-81ee-cc70b9056219" />

from category_encoders import TargetEncoder

te= TargetEncoder()

CC= df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC= pd.concat([CC,new],axis=1)

CC

<img width="768" height="471" alt="image" src="https://github.com/user-attachments/assets/c9cde337-5e8d-4e05-b857-41f7c999bc74" />

import pandas as pd 

import numpy as np

from scipy import stats

df= pd.read_csv("/content/Data_to_Transform.csv")

df

<img width="1064" height="507" alt="image" src="https://github.com/user-attachments/assets/52d7ddd8-df7e-4b67-8ea6-d48090aed175" />

df.skew()

<img width="437" height="246" alt="image" src="https://github.com/user-attachments/assets/43fd1e93-cca8-4d65-a4b6-8865727ccb9d" />

np.log(df["Highly Positive Skew"])

<img width="407" height="564" alt="image" src="https://github.com/user-attachments/assets/79da3e9b-f75d-46ec-8495-7d34a3bc83cb" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="408" height="511" alt="image" src="https://github.com/user-attachments/assets/0f0fd8ae-af82-4241-a9ef-127947938ae0" />

np.sqrt(df["Highly Positive Skew"])

<img width="361" height="518" alt="image" src="https://github.com/user-attachments/assets/a2c9db65-ec30-4bb2-ac2f-d267a23fbf7b" />

np.square(df["Highly Positive Skew"])

<img width="387" height="520" alt="image" src="https://github.com/user-attachments/assets/480bd3df-602f-4e40-a085-704ceeed7d26" />

df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])

df

<img width="1382" height="524" alt="image" src="https://github.com/user-attachments/assets/2e0895d5-739f-49e5-86d8-685757eb369d" />

df.skew()

<img width="484" height="301" alt="image" src="https://github.com/user-attachments/assets/ee22f7cb-8359-467e-b65c-16e95251d0d8" />

df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

<img width="536" height="353" alt="image" src="https://github.com/user-attachments/assets/a3b228e1-0401-46a6-a98f-7dea7538e2f8" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])

df

<img width="1453" height="586" alt="image" src="https://github.com/user-attachments/assets/9e892cfe-6ba7-4692-834d-747a9c0e68b3" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="773" height="556" alt="image" src="https://github.com/user-attachments/assets/1790790a-b770-43ef-8d54-661256aaefee" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

<img width="747" height="524" alt="image" src="https://github.com/user-attachments/assets/99847609-2be2-4789-a0b0-418cfbe6f405" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="708" height="541" alt="image" src="https://github.com/user-attachments/assets/0a79ae29-d4cf-4d73-ad88-5bd286692355" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()

<img width="739" height="550" alt="image" src="https://github.com/user-attachments/assets/eb933090-7dcd-4e17-b70b-ab425cc854e4" />

dt =pd.read_csv("titanic_dataset.csv")

dt

dt=pd.read_csv("titanic_dataset.csv")

dt

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt['Age'],line='45') 

plt.show()

<img width="708" height="523" alt="image" src="https://github.com/user-attachments/assets/b3b8e9e1-f5f7-4756-86fc-69ae667452d9" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()

<img width="706" height="527" alt="image" src="https://github.com/user-attachments/assets/32563f55-d00b-4ecb-964e-b08cf2b11b0e" />

















# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
