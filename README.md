- 👋 Hi, I’m @sourabhpu
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
sourabhpu/sourabhpu is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
#supervised learning
#import librarays 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#read dataset
import pandas as pd
df = pd.read_csv("C:\\Users\\ACER\\OneDrive\\Desktop\\mcdonalde\\mcdonaldata.csv")
df



df.describe()

	Unnamed: 0 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium
count 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000 	141.000000
mean 	70.000000 	7.493333 	10.060355 	5.000099 	1.108865 	26.321128 	30.770851 	15.409504 	10.336950 	362.918809
std 	40.847277 	8.336949 	10.435455 	4.898097 	7.319814 	50.348006 	20.664969 	15.674007 	14.283388 	477.792553
min 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000
25% 	35.000000 	0.650000 	0.460000 	0.330000 	0.070000 	1.470000 	15.630000 	2.280000 	0.000000 	41.990000
50% 	70.000000 	4.790000 	7.770000 	4.270000 	0.150000 	8.390000 	29.880000 	9.160000 	3.640000 	150.900000
75% 	105.000000 	10.880000 	14.160000 	7.280000 	0.250000 	31.110000 	45.390000 	26.950000 	19.230000 	530.540000
max 	140.000000 	39.470000 	45.180000 	20.460000 	75.260000 	302.610000 	93.840000 	64.220000 	64.220000 	2399.490000

df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 141 entries, 0 to 140
Data columns (total 14 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  141 non-null    int64  
 1   item        141 non-null    object 
 2   servesize   141 non-null    object 
 3   calories    141 non-null    object 
 4   protien     141 non-null    float64
 5   totalfat    141 non-null    float64
 6   satfat      141 non-null    float64
 7   transfat    141 non-null    float64
 8   cholestrol  141 non-null    float64
 9   carbs       141 non-null    float64
 10  sugar       141 non-null    float64
 11  addedsugar  141 non-null    float64
 12  sodium      141 non-null    float64
 13  menu        141 non-null    object 
dtypes: float64(9), int64(1), object(4)
memory usage: 15.5+ KB

df.corr()

C:\Users\ACER\AppData\Local\Temp\ipykernel_17760\1134722465.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
  df.corr()

	Unnamed: 0 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium
Unnamed: 0 	1.000000 	-0.407651 	-0.376072 	-0.342279 	-0.096239 	-0.230396 	-0.076254 	0.432428 	0.477586 	-0.343108
protien 	-0.407651 	1.000000 	0.871684 	0.701758 	0.079694 	0.590339 	0.415551 	-0.287476 	-0.319260 	0.909905
totalfat 	-0.376072 	0.871684 	1.000000 	0.844072 	0.062986 	0.421340 	0.535279 	-0.226667 	-0.282666 	0.859849
satfat 	-0.342279 	0.701758 	0.844072 	1.000000 	0.212726 	0.362520 	0.531387 	-0.047899 	-0.174590 	0.614493
transfat 	-0.096239 	0.079694 	0.062986 	0.212726 	1.000000 	0.034111 	-0.110521 	-0.056378 	-0.067910 	0.029505
cholestrol 	-0.230396 	0.590339 	0.421340 	0.362520 	0.034111 	1.000000 	0.152886 	-0.206585 	-0.225116 	0.475366
carbs 	-0.076254 	0.415551 	0.535279 	0.531387 	-0.110521 	0.152886 	1.000000 	0.506368 	0.455133 	0.493693
sugar 	0.432428 	-0.287476 	-0.226667 	-0.047899 	-0.056378 	-0.206585 	0.506368 	1.000000 	0.911110 	-0.299733
addedsugar 	0.477586 	-0.319260 	-0.282666 	-0.174590 	-0.067910 	-0.225116 	0.455133 	0.911110 	1.000000 	-0.265877
sodium 	-0.343108 	0.909905 	0.859849 	0.614493 	0.029505 	0.475366 	0.493693 	-0.299733 	-0.265877 	1.000000

df.notnull()

	Unnamed: 0 	item 	servesize 	calories 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium 	menu
0 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
1 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
2 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
3 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
4 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
136 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
137 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
138 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
139 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True
140 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True 	True

141 rows × 14 columns

df.nunique()

Unnamed: 0    141
item          141
servesize     120
calories      140
protien       117
totalfat      111
satfat        106
transfat       43
cholestrol    115
carbs         134
sugar         124
addedsugar     79
sodium        139
menu            7
dtype: int64

df.tail()

	Unnamed: 0 	item 	servesize 	calories 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium 	menu
136 	136 	Tomato Ketchup Sachets 	8 	11.23 	0.08 	23.45 	0.38 	0.25 	0.08 	2.63 	2.33 	1.64 	414.71 	condiments
137 	137 	Maple Syrup 	3 	86.4 	0.00 	0.00 	0.00 	0.40 	0.30 	21.60 	16.20 	5.34 	71.05 	condiments
138 	138 	Cheese Slice 	14 	51.03 	3.06 	3.99 	0.00 	0.00 	13.43 	0.72 	0.54 	0.00 	15.00 	condiments
139 	139 	Sweet Corn 	40 	45.08 	1.47 	1.00 	2.89 	0.01 	2.00 	7.55 	2.54 	0.00 	178.95 	condiments
140 	140 	Mixed Fruit Beverage 	180 	72.25 	0.65 	0.02 	0.22 	0.04 	0.01 	18.00 	16.83 	0.00 	0.04 	condiments

df['protien'].value_counts()

0.00     16
1.52      3
0.52      3
2.05      2
5.67      2
         ..
11.49     1
9.58      1
10.22     1
4.48      1
0.65      1
Name: protien, Length: 117, dtype: int64

df['protien'].sum()

1056.56

df['totalfat'].sum()

1418.5100000000002

df

	Unnamed: 0 	item 	servesize 	calories 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium 	menu
0 	0 	McVeggie Burger 	168 	402 	10.24 	13.83 	5.34 	0.16 	2.49 	56.54 	7.90 	4.49 	706.13 	regular
1 	1 	McAloo Tikki Burger 	146 	339 	8.50 	11.31 	4.27 	0.20 	1.47 	5.27 	7.05 	4.07 	545.34 	regular
2 	2 	McSpicy Paneer Burger 	199 	652 	20.29 	39.45 	17.12 	0.18 	21.85 	52.33 	8.35 	5.27 	1074.58 	regular
3 	3 	Spicy Paneer Wrap 	250 	674 	20.96 	39.10 	19.73 	0.26 	40.93 	59.27 	3.50 	1.08 	1087.46 	regular
4 	4 	American Veg Burger 	177 	512 	15.30 	23.45 	10.51 	0.17 	25.24 	56.96 	7.85 	4.76 	1051.24 	regular
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
136 	136 	Tomato Ketchup Sachets 	8 	11.23 	0.08 	23.45 	0.38 	0.25 	0.08 	2.63 	2.33 	1.64 	414.71 	condiments
137 	137 	Maple Syrup 	3 	86.4 	0.00 	0.00 	0.00 	0.40 	0.30 	21.60 	16.20 	5.34 	71.05 	condiments
138 	138 	Cheese Slice 	14 	51.03 	3.06 	3.99 	0.00 	0.00 	13.43 	0.72 	0.54 	0.00 	15.00 	condiments
139 	139 	Sweet Corn 	40 	45.08 	1.47 	1.00 	2.89 	0.01 	2.00 	7.55 	2.54 	0.00 	178.95 	condiments
140 	140 	Mixed Fruit Beverage 	180 	72.25 	0.65 	0.02 	0.22 	0.04 	0.01 	18.00 	16.83 	0.00 	0.04 	condiments

141 rows × 14 columns

#defining x and y

x=df.iloc[:,2:8]#independent variable 

y=df.iloc[:,7:13]#dependent variable

x.shape,y.shape

((141, 6), (141, 6))

x.head()

	servesize 	calories 	protien 	totalfat 	satfat 	transfat
0 	168 	402 	10.24 	13.83 	5.34 	0.16
1 	146 	339 	8.50 	11.31 	4.27 	0.20
2 	199 	652 	20.29 	39.45 	17.12 	0.18
3 	250 	674 	20.96 	39.10 	19.73 	0.26
4 	177 	512 	15.30 	23.45 	10.51 	0.17

y.head()

	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium
0 	0.16 	2.49 	56.54 	7.90 	4.49 	706.13
1 	0.20 	1.47 	5.27 	7.05 	4.07 	545.34
2 	0.18 	21.85 	52.33 	8.35 	5.27 	1074.58
3 	0.26 	40.93 	59.27 	3.50 	1.08 	1087.46
4 	0.17 	25.24 	56.96 	7.85 	4.76 	1051.24

df.corr()

C:\Users\ACER\AppData\Local\Temp\ipykernel_17760\1134722465.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
  df.corr()

	Unnamed: 0 	protien 	totalfat 	satfat 	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium
Unnamed: 0 	1.000000 	-0.407651 	-0.376072 	-0.342279 	-0.096239 	-0.230396 	-0.076254 	0.432428 	0.477586 	-0.343108
protien 	-0.407651 	1.000000 	0.871684 	0.701758 	0.079694 	0.590339 	0.415551 	-0.287476 	-0.319260 	0.909905
totalfat 	-0.376072 	0.871684 	1.000000 	0.844072 	0.062986 	0.421340 	0.535279 	-0.226667 	-0.282666 	0.859849
satfat 	-0.342279 	0.701758 	0.844072 	1.000000 	0.212726 	0.362520 	0.531387 	-0.047899 	-0.174590 	0.614493
transfat 	-0.096239 	0.079694 	0.062986 	0.212726 	1.000000 	0.034111 	-0.110521 	-0.056378 	-0.067910 	0.029505
cholestrol 	-0.230396 	0.590339 	0.421340 	0.362520 	0.034111 	1.000000 	0.152886 	-0.206585 	-0.225116 	0.475366
carbs 	-0.076254 	0.415551 	0.535279 	0.531387 	-0.110521 	0.152886 	1.000000 	0.506368 	0.455133 	0.493693
sugar 	0.432428 	-0.287476 	-0.226667 	-0.047899 	-0.056378 	-0.206585 	0.506368 	1.000000 	0.911110 	-0.299733
addedsugar 	0.477586 	-0.319260 	-0.282666 	-0.174590 	-0.067910 	-0.225116 	0.455133 	0.911110 	1.000000 	-0.265877
sodium 	-0.343108 	0.909905 	0.859849 	0.614493 	0.029505 	0.475366 	0.493693 	-0.299733 	-0.265877 	1.000000

#split dependent and independent var. 

#split x and y in trained data set and test data set

from sklearn.model_selection  import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

((98, 6), (43, 6), (98, 6), (43, 6))

#Linear regression

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

LinearRegression()

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

y_pred=model.predict(x_test)

y_pred

​

array([[-1.70465352e-12, -6.87574914e+00,  3.33116331e+01,
         3.02416850e+01,  2.60526624e+01,  3.74897257e+01],
       [ 1.00000000e-01,  2.87064345e+00,  6.21532447e+01,
         2.50384715e+01,  1.75485960e+01,  5.67901132e+02],
       [ 1.30000000e-01,  2.26282743e+01,  4.51973326e+01,
         1.58990600e+01,  1.17823051e+01,  6.00747596e+02],
       [ 1.70000000e-01,  1.05550754e+02,  3.39720917e+01,
         1.45909773e+00, -4.14349410e+00,  1.06592123e+03],
       [ 3.80000000e-01,  4.73395702e+01,  2.50181984e+01,
         1.70926506e+01,  9.91002114e+00,  3.40157925e+02],
       [ 1.50000000e-01,  5.28027504e+00,  4.44102999e+01,
         1.83791173e+01,  1.23996689e+01,  3.80268944e+02],
       [ 8.00000000e-02,  1.87055290e+00,  4.55924657e+01,
         1.61251509e+01,  1.07346204e+01,  4.58460927e+02],
       [ 2.00000000e-01,  2.68956074e+01,  4.45347454e+01,
         1.72822269e+01,  1.29337973e+01,  5.52202851e+02],
       [ 5.00000000e-02,  5.09090629e+00,  9.63715141e+00,
         1.70990493e+01,  1.41370882e+01, -4.99259426e+01],
       [ 1.50000000e-01,  1.70866376e+01,  3.10317936e+01,
         2.23900672e+01,  1.72545830e+01,  1.80368306e+02],
       [ 2.80000000e-01,  4.12770611e+00,  6.42296552e+00,
         1.37576981e+01,  1.08564310e+01, -8.31286730e+01],
       [ 4.00000000e-01,  6.97585363e+00,  1.81614036e+01,
         1.01127839e+01,  6.99260686e+00,  1.71245084e+01],
       [ 3.20000000e-01,  2.95955080e+01,  4.32617091e+01,
         2.40894182e+01,  1.74635146e+01,  3.65441896e+02],
       [-1.68021085e-12, -8.07741192e+00,  3.84588225e+01,
         3.26232072e+01,  2.81860390e+01,  6.58128459e+01],
       [ 1.50000000e-01,  7.32337271e+01,  3.53102518e+01,
         4.44278983e-01, -1.92983774e+00,  1.09583113e+03],
       [ 2.80000000e-01,  4.92642840e+00,  6.33489912e+00,
         1.36225053e+01,  1.07363356e+01, -7.82197796e+01],
       [ 1.80000000e-01,  8.75832230e+01,  3.83883419e+01,
         4.36650216e+00,  6.78475396e-01,  1.05962854e+03],
       [ 1.30000000e-01,  1.39645297e+01,  5.00742844e+01,
         2.55670131e+01,  1.92461685e+01,  3.27711534e+02],
       [ 2.60000000e-01,  1.61033769e+02,  3.44990756e+01,
        -1.21333214e+01, -1.26319541e+01,  1.91379792e+03],
       [ 2.20000000e-01,  1.09517572e+02,  9.97534363e+00,
        -7.87471416e+00, -1.13732206e+01,  8.51045949e+02],
       [ 1.50000000e-01,  2.94761813e+01,  3.05928959e+01,
         1.61107931e+01,  1.08100761e+01,  2.62308294e+02],
       [ 4.60000000e-01, -1.88493510e-01,  9.19977600e+00,
         2.08700633e+01,  1.76469140e+01, -8.89277866e+01],
       [-1.69243219e-12, -7.47658053e+00,  3.58852278e+01,
         3.14324461e+01,  2.71193507e+01,  5.16512858e+01],
       [-2.98975727e-12, -1.40248065e+01,  4.66509169e+01,
         4.14540192e+01,  3.65124405e+01,  8.76030109e+01],
       [ 4.10000000e+01,  3.11017333e+01,  2.12086176e+01,
         1.75892303e+01,  1.56314192e+01,  5.76073895e+02],
       [ 1.00000000e-02,  1.05472979e+01,  3.60960300e+00,
         3.57678180e+00,  1.09804500e+00, -6.52921099e+01],
       [ 2.50000000e-01,  1.17203958e+02,  4.77726276e+01,
        -5.22516435e+00, -6.04950073e+00,  1.86269261e+03],
       [ 6.00000000e-02,  2.57236333e+00,  1.15793784e+01,
         2.15103946e+01,  1.83837559e+01, -5.02647934e+01],
       [ 1.60000000e-01,  1.07420921e+00,  2.26264586e+01,
         1.98685459e+01,  1.62607435e+01,  1.05473525e+01],
       [-2.95465646e-12, -1.57504501e+01,  5.40425203e+01,
         4.48739958e+01,  3.95760684e+01,  1.28276329e+02],
       [ 1.60000000e-01,  5.91431898e+01,  4.07016969e+01,
         9.64940917e+00,  6.20346822e+00,  8.54114992e+02],
       [ 1.10000000e-01,  4.94924759e+01,  2.59314271e+01,
         4.27542598e+00,  1.88903997e+00,  6.44858242e+02],
       [ 3.00000000e-01,  3.96752605e+01,  2.03140044e+01,
         1.41499884e+01,  8.00206142e+00,  2.54109992e+02],
       [ 1.80000000e-01,  4.42618893e+01,  3.71247989e+01,
         1.37603349e+01,  7.56348426e+00,  4.71817227e+02],
       [ 1.60000000e-01,  3.13146373e+01,  3.70789301e+01,
         1.88957167e+01,  1.32476202e+01,  3.23615888e+02],
       [ 3.10000000e-01,  4.06843683e+01,  2.09200708e+01,
         1.45254358e+01,  8.22677753e+00,  2.64978888e+02],
       [ 1.60000000e-01,  1.37296102e+01,  5.35884099e+01,
         3.39640469e+01,  2.17839133e+01,  1.35363863e+02],
       [ 4.00000000e-02,  6.68227410e+00,  1.70920492e+01,
         1.52626519e+01,  1.18920067e+01, -1.59813618e+00],
       [ 3.30000000e-01,  2.92453384e+00,  7.21280603e+00,
         1.57975258e+01,  1.28053059e+01, -8.45799024e+01],
       [ 1.00000000e-01,  1.34609164e+01,  3.33894790e+01,
         1.78360090e+01,  1.25690955e+01,  1.83514035e+02],
       [ 1.00000000e-02,  2.02300758e+01,  9.32935991e+00,
         7.79297462e+00,  3.02627496e+00, -7.55733285e+01],
       [ 9.00000000e-02,  3.50325633e+01,  2.77974159e+01,
         8.97325487e+00,  3.25347917e+00,  3.77553812e+02],
       [ 1.30000000e-01,  1.48752097e+01,  3.12461970e+01,
         1.75865550e+01,  1.15727349e+01,  1.51644642e+02]])

#evalution

'''

1.mean_absolute_error

2.mean_squared_error

3.r2_score'''

print(model.score(x_train,y_train))

print(model.score(x_test,y_test))

print('.........')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print(mean_absolute_error(y_test,y_pred))

print(mean_squared_error(y_test,y_pred))

print(r2_score(y_test,y_pred))

0.707398134619575
0.69412686107594
.........
26.1847857179601
4044.5629567952346
0.69412686107594

x_test.head(2),x_test.shape,y_test.shape

(    servesize calories  protien  totalfat  satfat  transfat
 122       394    137.6     0.00      0.00    0.00       0.0
 28       154       449     6.76     20.77    9.95       0.1,
 (43, 6),
 (43, 6))

​

plt.scatter(x_test.iloc[:,1],y_test.iloc[:,1])

plt.plot(x_test.iloc[:,1],y_pred[:,1])

plt.grid()

plt.show()

#multi linear regre.

​

x=df.iloc[:,3:12]

y=df.iloc[:,12:13]

x.shape,y.shape

((141, 9), (141, 1))

x.head(2),y.head(2)

(  calories  protien  totalfat  satfat  transfat  cholestrol  carbs  sugar  \
 0      402    10.24     13.83    5.34      0.16        2.49  56.54   7.90   
 1      339     8.50     11.31    4.27      0.20        1.47   5.27   7.05   
 
    addedsugar  
 0        4.49  
 1        4.07  ,
    sodium
 0  706.13
 1  545.34)

#difine x and y

# x=df[['SepalLengthCm','SepalWidthCm']]

# y=df['PetalLengthCm']

# x.shape,y.shape

​

#split dependent and independent var. 

#split x and y in trained data set and test data set

# from sklearn.model_selection  import train_test_spl/it

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25)

# x_train.shape,x_test.shape,y_train.shape,y_test.shape

# Polynomial tree regresor

from sklearn.preprocessing import PolynomialFeatures

model=PolynomialFeatures(degree=2)

x_train_pr=model.fit_transform(x_train)

x_test_pr=model.fit_transform(x_test)

# x_test_pr[:5]

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train_pr,y_train)

y_pred=model.predict(x_test_pr)

y_pred[:5],y_test.values[:5]

(array([[ 1.85331947e-13, -2.01935959e+01,  3.21428365e+01,
          3.04762254e+01,  3.36137199e+01,  2.52713257e+02],
        [ 1.00000000e-01,  9.37170334e+01,  5.20691906e+01,
          1.97905946e+01, -4.34299040e+00, -1.67206482e+02],
        [ 1.30000000e-01,  4.60052872e+01,  4.85846626e+01,
          1.11669660e+01,  5.35127831e+00,  6.14546814e+02],
        [ 1.70000000e-01,  1.76445206e+02,  3.90921255e+01,
          1.13408108e+01,  2.63170719e+00,  9.10607849e+02],
        [ 3.80000000e-01,  1.27682480e+02,  2.39986339e+01,
          2.87122755e+01, -2.08900309e+00, -2.95061157e+02]]),
 array([[0.00000e+00, 0.00000e+00, 3.44000e+01, 3.44000e+01, 3.44000e+01,
         4.98900e+01],
        [1.00000e-01, 1.54000e+00, 5.41600e+01, 7.70000e-01, 0.00000e+00,
         3.06290e+02],
        [1.30000e-01, 1.51000e+00, 4.79000e+01, 3.64000e+00, 3.49000e+00,
         5.48790e+02],
        [1.70000e-01, 7.12300e+01, 3.74500e+01, 7.64000e+00, 3.84000e+00,
         1.39617e+03],
        [3.80000e-01, 3.89500e+01, 2.07700e+01, 1.54000e+01, 0.00000e+00,
         1.65360e+02]]))

print('-----------Accuracy score-------------')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from math import sqrt

print(mean_absolute_error(y_test,y_pred))

print(mean_squared_error(y_test,y_pred))

mse =mean_squared_error(y_test,y_pred)

print(r2_score(y_test,y_pred))

-----------Accuracy score-------------
94.64951901299094
139939.9785677696
-26.023270217256055

x_test.head(2)

	servesize 	calories 	protien 	totalfat 	satfat 	transfat
122 	394 	137.6 	0.00 	0.00 	0.00 	0.0
28 	154 	449 	6.76 	20.77 	9.95 	0.1

# x_test_pr[:5,4:5],x_test_pr[:5,4:5]

#difine x and y

# x=df[['protien','calories']]

# y=df['totalfat']

x.shape,y.shape

((141, 9), (141, 1))

x_test.head(2)

	servesize 	calories 	protien 	totalfat 	satfat 	transfat
122 	394 	137.6 	0.00 	0.00 	0.00 	0.0
28 	154 	449 	6.76 	20.77 	9.95 	0.1

y_test.head(2)

	transfat 	cholestrol 	carbs 	sugar 	addedsugar 	sodium
122 	0.0 	0.00 	34.40 	34.40 	34.4 	49.89
28 	0.1 	1.54 	54.16 	0.77 	0.0 	306.29

​

plt.figure(figsize=(25,3))

plt.subplot(1,2,1)

plt.scatter(x_test['calories'],y_test['transfat'])

plt.plot(x_test['calories'],y_pred[:,1],'red')

plt.show()

plt.figure(figsize=(10,3))

plt.subplot(1,2,2)

ax=plt.axes(projection='3d')

ax.bar(x_test_pr[:,1],x_test_pr[:,2],y_test['transfat'])

ax.bar(x_test_pr[:,1],x_test_pr[:,2],y_pred[:,1])

plt.show()

ax=plt.axes(projection='3d')

ax.bar(x_test_pr['calories'],x_test_pr['protien'],y_test)

ax.bar(x_test_pr['calories'],x_test_pr['protien'],y_pred)

plt.show()

#desicion tree

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor()

print(dtr.fit(x_train,y_train))

y_pred_dtr=dtr.predict(x_test)

print(y_pred_dtr[:5])

print(y_test.values[:5])

print('------accuracy score--------')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from math import sqrt

print('mean_absolute_error',mean_absolute_error(y_test,y_pred_dtr))

print('mean_squared_error',mean_squared_error(y_test,y_pred_dtr))

mse=mean_squared_error(y_test,y_pred_dtr)

print('r2_score',r2_score(y_test,y_pred_dtr))

print('MODEL SCORE',dtr.score(x_test,y_test))

print(sqrt(mse))

ax = plt.axes(projection ='3d')

ax.scatter3D(x_test_pr[:,1],x_test_pr[:,2],y_test['transfat'])

ax.scatter3D(x_test_pr[:,1],x_test_pr[:,2],y_pred[:,1],'red')

plt.show()

DecisionTreeRegressor()
[[0.00000e+00 0.00000e+00 3.23700e+01 3.23700e+01 3.23700e+01 7.67100e+01]
 [1.70000e-01 2.52400e+01 5.69600e+01 7.85000e+00 4.76000e+00 1.05124e+03]
 [2.40000e-01 9.45000e+00 4.63600e+01 4.53000e+00 1.15000e+00 5.79600e+02]
 [2.70000e-01 8.76300e+01 5.70600e+01 8.92000e+00 1.08000e+00 1.15238e+03]
 [2.20000e-01 9.36900e+00 7.25100e+01 5.51400e+01 4.43500e+01 3.32600e+02]]
[[0.00000e+00 0.00000e+00 3.44000e+01 3.44000e+01 3.44000e+01 4.98900e+01]
 [1.00000e-01 1.54000e+00 5.41600e+01 7.70000e-01 0.00000e+00 3.06290e+02]
 [1.30000e-01 1.51000e+00 4.79000e+01 3.64000e+00 3.49000e+00 5.48790e+02]
 [1.70000e-01 7.12300e+01 3.74500e+01 7.64000e+00 3.84000e+00 1.39617e+03]
 [3.80000e-01 3.89500e+01 2.07700e+01 1.54000e+01 0.00000e+00 1.65360e+02]]
------accuracy score--------
mean_absolute_error 26.56427131782945
mean_squared_error 6784.763667914728
r2_score 0.22610954781613293
MODEL SCORE 0.22610954781613293
82.3696768714964

y.shape,x.shape

((141, 1), (141, 9))

​

​

​

​

(98, 6)
(98, 6)

	servesize 	calories 	protien 	totalfat 	satfat 	transfat
122 	394 	137.6 	0.00 	0.00 	0.00 	0.00
28 	154 	449 	6.76 	20.77 	9.95 	0.10
14 	138 	357 	8.64 	14.02 	4.84 	0.13
104 	195 	457.94 	24.43 	22.65 	11.56 	0.17
54 	375 	232.2 	11.14 	12.82 	9.43 	0.38

x_train,y_train.values[:,1]

(    servesize calories  protien  totalfat  satfat  transfat
 7         87       228     5.45     11.44    5.72      0.09
 24        87       246    15.26     18.57   17.12     75.26
 25       145       411    25.43     28.54    0.15      0.08
 91        317   398.19     5.67     12.77   11.38      0.20
 49      201.5   125.25     6.02      7.01    5.15      0.20
 ..        ...      ...      ...       ...     ...       ...
 121       299     99.6     0.00      0.00    0.00      0.00
 97    132.08    156.14     2.05      2.36    1.74      0.10
 29       114       204     3.97      7.15    3.39      0.13
 129    286.79   145.16     1.52      1.75    1.28      0.25
 43         64   140.29     1.93      7.32    3.42      0.06
 
 [98 rows x 6 columns],
 array([5.1700e+00, 4.5150e+01, 6.7000e+00, 1.0890e+01, 2.1270e+01,
        2.4660e+01, 2.9000e-01, 2.5240e+01, 6.0000e+00, 4.7000e+00,
        3.3000e+00, 2.4900e+00, 8.7630e+01, 7.3110e+01, 4.0930e+01,
        1.5700e+00, 9.3690e+00, 2.9520e+01, 1.2270e+01, 8.7400e+00,
        9.7600e+00, 1.8400e+00, 2.7000e-01, 2.5470e+01, 0.0000e+00,
        2.4430e+01, 3.1110e+01, 2.1309e+02, 9.4200e+00, 8.0000e+00,
        1.3430e+01, 0.0000e+00, 5.0000e-02, 2.8140e+01, 4.7000e+00,
        0.0000e+00, 4.6000e-01, 4.3680e+01, 2.7000e-01, 3.2830e+01,
        0.0000e+00, 3.0050e+01, 5.8500e+00, 2.1850e+01, 9.8900e+00,
        5.5480e+01, 4.8740e+01, 6.4190e+01, 3.7750e+01, 9.4500e+00,
        0.0000e+00, 3.5670e+01, 3.0100e+01, 9.2300e+00, 0.0000e+00,
        5.3020e+01, 0.0000e+00, 3.6990e+01, 2.7900e+00, 3.6670e+01,
        8.0000e-02, 0.0000e+00, 4.0000e-01, 3.1320e+01, 0.0000e+00,
        7.8520e+01, 8.3900e+00, 0.0000e+00, 7.7000e-01, 5.7100e+00,
        4.8500e+00, 1.6500e+00, 2.5000e-01, 9.9900e+00, 3.6480e+01,
        2.2030e+01, 4.8000e+00, 4.5600e+00, 8.1000e+00, 3.0261e+02,
        4.0000e-01, 2.3330e+02, 2.1261e+02, 3.3210e+01, 4.7500e+00,
        3.6550e+01, 6.2700e+00, 2.9070e+01, 1.3300e+00, 3.6190e+01,
        8.3900e+00, 6.1900e+00, 3.3000e+00, 0.0000e+00, 6.5500e+00,
        9.7000e-01, 4.7000e+00, 6.4000e-01]))

#support vector machine

from sklearn.svm import SVR

print(' svc - poly')

svcp=SVR(kernel='poly', degree=2, C=.1)

svcp.fit(x_train,y_train.values[:,1])

y_pred_svcp=svcp.predict(x_test)

print(f'predicted_y-{y_pred_svcp[:5]}') 

print(f'actual_y-{y_test.values[:5]}')

print('------accuracy score--------')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from math import sqrt

print('mean_absolute_error',mean_absolute_error(y_test.values[:,1],y_pred_svcp))

print('mean_squared_error',mean_squared_error(y_test.values[:,1],y_pred_svcp))

mse=mean_squared_error(y_test.values[:,1],y_pred_svcp)

print('r2_score',r2_score(y_test.values[:,1],y_pred_svcp))

print('MODEL SCORE',svcp.score(x_test,y_test.values[:,1]))

print(sqrt(mse))

​

ax = plt.axes(projection ='3d')

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_test.values[:,1])

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_pred[:,1],'red')

plt.show()

 svc - poly
predicted_y-[ 6.16390936 13.29526694 10.44014463 13.96158748  8.16635091]
actual_y-[[0.00000e+00 0.00000e+00 3.44000e+01 3.44000e+01 3.44000e+01 4.98900e+01]
 [1.00000e-01 1.54000e+00 5.41600e+01 7.70000e-01 0.00000e+00 3.06290e+02]
 [1.30000e-01 1.51000e+00 4.79000e+01 3.64000e+00 3.49000e+00 5.48790e+02]
 [1.70000e-01 7.12300e+01 3.74500e+01 7.64000e+00 3.84000e+00 1.39617e+03]
 [3.80000e-01 3.89500e+01 2.07700e+01 1.54000e+01 0.00000e+00 1.65360e+02]]
------accuracy score--------
mean_absolute_error 24.74767946695295
mean_squared_error 3036.4927454369813
r2_score -0.08010031743722212
MODEL SCORE -0.08010031743722212
55.10438045597628

from sklearn.svm import SVR

print(' svc - linear')

svcp=SVR(kernel='linear', degree=2, C=.1)

svcp.fit(x_train,y_train.values[:,1])

y_pred_svcp=svcp.predict(x_test)

print(f'predicted_y-{y_pred_svcp[:5]}') 

print(f'actual_y-{y_test.values[:5]}')

print('------accuracy score--------')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from math import sqrt

print('mean_absolute_error',mean_absolute_error(y_test.values[:,1],y_pred_svcp))

print('mean_squared_error',mean_squared_error(y_test.values[:,1],y_pred_svcp))

mse=mean_squared_error(y_test.values[:,1],y_pred_svcp)

print('r2_score',r2_score(y_test.values[:,1],y_pred_svcp))

print('MODEL SCORE',svcp.score(x_test,y_test.values[:,1]))

print(sqrt(mse))

​

ax = plt.axes(projection ='3d')

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_test.values[:,1])

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_pred[:,1],'red')

plt.show()

 svc - linear
predicted_y-[ 0.22502206  6.99902193 14.83334474 61.13137091 30.88037745]
actual_y-[[0.00000e+00 0.00000e+00 3.44000e+01 3.44000e+01 3.44000e+01 4.98900e+01]
 [1.00000e-01 1.54000e+00 5.41600e+01 7.70000e-01 0.00000e+00 3.06290e+02]
 [1.30000e-01 1.51000e+00 4.79000e+01 3.64000e+00 3.49000e+00 5.48790e+02]
 [1.70000e-01 7.12300e+01 3.74500e+01 7.64000e+00 3.84000e+00 1.39617e+03]
 [3.80000e-01 3.89500e+01 2.07700e+01 1.54000e+01 0.00000e+00 1.65360e+02]]
------accuracy score--------
mean_absolute_error 13.152585254652776
mean_squared_error 1819.5617723896255
r2_score 0.35276998408509563
MODEL SCORE 0.35276998408509563
42.65632159937874

from sklearn.svm import SVR

print(' svc - rbf')

svcp=SVR(kernel='rbf', degree=2, C=.1)

svcp.fit(x_train,y_train.values[:,1])

y_pred_svcp=svcp.predict(x_test)

print(f'predicted_y-{y_pred_svcp[:5]}') 

print(f'actual_y-{y_test.values[:5]}')

print('------accuracy score--------')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from math import sqrt

print('mean_absolute_error',mean_absolute_error(y_test.values[:,1],y_pred_svcp))

print('mean_squared_error',mean_squared_error(y_test.values[:,1],y_pred_svcp))

mse=mean_squared_error(y_test.values[:,1],y_pred_svcp)

print('r2_score',r2_score(y_test.values[:,1],y_pred_svcp))

print('MODEL SCORE',svcp.score(x_test,y_test.values[:,1]))

print(sqrt(mse))

​

ax = plt.axes(projection ='3d')

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_test.values[:,1])

ax.scatter3D(x_test.iloc[:,1],x_test.iloc[:,2],y_pred[:,1],'red')

plt.show()

 svc - rbf
predicted_y-[7.34425482 9.38201244 8.98876334 9.44311569 7.97377415]
actual_y-[[0.00000e+00 0.00000e+00 3.44000e+01 3.44000e+01 3.44000e+01 4.98900e+01]
 [1.00000e-01 1.54000e+00 5.41600e+01 7.70000e-01 0.00000e+00 3.06290e+02]
 [1.30000e-01 1.51000e+00 4.79000e+01 3.64000e+00 3.49000e+00 5.48790e+02]
 [1.70000e-01 7.12300e+01 3.74500e+01 7.64000e+00 3.84000e+00 1.39617e+03]
 [3.80000e-01 3.89500e+01 2.07700e+01 1.54000e+01 0.00000e+00 1.65360e+02]]
------accuracy score--------
mean_absolute_error 26.148309207485667
mean_squared_error 3180.9552154640755
r2_score -0.1314865622976802
MODEL SCORE -0.1314865622976802
56.39995758388543

