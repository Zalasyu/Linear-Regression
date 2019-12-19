"""
Problem 1
The amounts m of a chemical compound that dissolved in 100 grams of water at
various temperatures T were recorded as in Table 1.
    a. Using the formulas in your textbook (page 149), calculate the intercept and slope of
        the regression line for the sample. (10 points)
    b. Graph the line on a scatter diagram and estimate the amount of chemical that will
        dissolve in 100 grams of water at 50ºC. (5 points)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(color_codes=True)
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from statistics import stdev

sns.set_style('darkgrid')
sns.set_context('presentation')

df = pd.DataFrame(
    {'Temperature (C)': [0, 15, 30, 45, 69, 75], 'Trial 1': [8, 12, 25, 31, 44, 48], 'Trail 2': [6, 10, 21, 33, 39, 51],
     'Trail 3': [8, 14, 24, 28, 42, 44]})
df = df.melt('Temperature (C)', var_name='cols', value_name='grams')
x = df['Temperature (C)']
y = df['grams']

model = sm.OLS(y, x).fit()  # OLS = Ordinary Least Squares
print(model.summary())

fig, axs = plt.subplots(2)
fig.suptitle('Homework 8 Graphs', fontsize=14, fontweight='bold')
ax1 = sns.regplot('Temperature (C)', 'grams', data=df, fit_reg=True, ax=axs[0])
axs[0].set_title('Temperature vs. Grams of Chemical Dissolved')


def find_linear(x, y):
    m = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x * x) - np.sum(x) ** 2)
    b = (np.sum(y) - m * np.sum(x)) / len(x)
    return m, b


def pearson_coef_and_pvalue(x, y):
    pearson_coef, p_value = stats.pearsonr(x, y)
    print(f'The pearson coefficient is :{pearson_coef}.\nThe p-value is: {p_value}')
    return pearson_coef, p_value


m1, b1 = find_linear(x, y)
print(f'The slope is: {m1:.2f}. The intercept is: {b1:.2f}\n\n')
plt.text(13, 95, f'y={b1:.2f}+{m1:.2f}*x')
plt.text(13, 86, f'R^2 = 0.981')


def predict(x, m, b):
    return m * x + b


predict50 = predict(50, m1, b1)
print(f' The amount of chemical dissolved at 50 degrees Celsius is:\n{predict50:.2f} grams\n\n')

plt.title('Temperature vs Dissolution')
print('\n\n__________________________________________________________________________________\n\n')
"""
Problem 2
The Statistics Consulting Center at Virginia Tech analyzed data on normal woodchucks
for the Department of Veterinary Medicine. The variables of interest were body weight
in grams and heart weight in grams, as reported in Table 2. It was desired to develop a
linear regression equation in order to determine if there is a significant linear
relationship between heart weight and total body weight.
    a. With heart weight as the independent variable and body weight as the dependent
        variable, fit a linear regression, using JASP. (5 points)
    b. What percent of the variation in the body weight is
        accounted for by difference in heart weight? (5 points)
    c. How confident are you that the linear regression for the
        sample is not just due to chance? (5 points)
"""
df2 = pd.DataFrame(
    {'Body Weight (kg)': [4.050, 2.465, 3.120, 5.700, 2.595, 3.640, 2.050, 4.235, 2.935, 4.975, 3.690, 2.800, 2.775,
                          2.170, 2.370, 2.055, 2.025, 2.645, 2.675],
     'Heart Weight (g)': [11.2, 12.4, 10.5, 13.2, 9.8, 11.0, 10.8, 10.4, 12.2, 11.2, 10.8, 14.2, 12.2,
                          10.0, 12.3, 12.5, 11.8, 16.0, 13.8]})

model2 = sm.OLS(df2['Body Weight (kg)'], df2['Heart Weight (g)']).fit()
print(model2.summary())

m2, b2 = find_linear(df2['Heart Weight (g)'], df2['Body Weight (kg)'])
print(f'The slope is: {m2:.2f}. The intercept is: {b2:.2f}\n\n')
plt.text(13, 25, f'y={b2:.2f}+{m2:.2f}*x')
plt.text(13, 16, f'R^2 = 0.885')

ax2 = sns.regplot('Heart Weight (g)', 'Body Weight (kg)', data=df2, fit_reg=True, ax=axs[1])
axs[1].set_title('Heart Weight (g) vs. Body Weight (kg)')

ax2.set_xlim([8, 15])
ax2.set_ylim([-12, 40])
plt.subplots_adjust(hspace=0.7)

print('b. The model explains 88.5% of the data variability. There is a strong correlation with thr associated variables.\n\n')
print('c. The F-Test\n From the OLS Regression Analysis Summary the Prob(F-statistic) is 6.77e-10, which is grossly '
      'lower than our signigicane level of 0.05.\nTherefore, we reject the null hypothesis that there is no '
      'association betweent he X variable (Heart Weight) and the Y variabel (Body Weight).')



plt.show()

"""
        Problem 1: ANALYSIS
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                  grams   R-squared (uncentered):                   0.981
Model:                            OLS   Adj. R-squared (uncentered):              0.980
Method:                 Least Squares   F-statistic:                              871.1
Date:                Wed, 27 Nov 2019   Prob (F-statistic):                    4.80e-16
Time:                        18:58:49   Log-Likelihood:                         -51.669
No. Observations:                  18   AIC:                                      105.3
Df Residuals:                      17   BIC:                                      106.2
Df Model:                           1                                                  
Covariance Type:            nonrobust                                                  
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
Temperature (C)     0.6435      0.022     29.514      0.000       0.597       0.689
==============================================================================
Omnibus:                        0.432   Durbin-Watson:                   1.143
Prob(Omnibus):                  0.806   Jarque-Bera (JB):                0.551
Skew:                          -0.224   Prob(JB):                        0.759
Kurtosis:                       2.270   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

The slope is: 0.54. The intercept is: 6.19

The amount of chemical dissolved at 50 degrees Celsius is:
33.01 grams




__________________________________________________________________________________

Problem 2: ANALYSIS
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:       Body Weight (kg)   R-squared (uncentered):                   0.885
Model:                            OLS   Adj. R-squared (uncentered):              0.879
Method:                 Least Squares   F-statistic:                              138.8
Date:                Wed, 27 Nov 2019   Prob (F-statistic):                    6.77e-10
Time:                        18:58:49   Log-Likelihood:                         -28.857
No. Observations:                  19   AIC:                                      59.71
Df Residuals:                      18   BIC:                                      60.66
Df Model:                           1                                                  
Covariance Type:            nonrobust                                                  
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Heart Weight (g)     0.2555      0.022     11.783      0.000       0.210       0.301
==============================================================================
Omnibus:                        2.296   Durbin-Watson:                   1.533
Prob(Omnibus):                  0.317   Jarque-Bera (JB):                1.867
Skew:                           0.666   Prob(JB):                        0.393
Kurtosis:                       2.236   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The slope is: -0.05. The intercept is: 3.67

b. The model explains 88.5% of the data variability. There is a strong correlation with thr associated variables.


c. The F-Test From the OLS Regression Analysis Summary the Prob(F-statistic) is 6.77e-10, which is grossly lower than 
our significance level of 0.05. Therefore, we reject the null hypothesis that there is no association between the X 
variable (Heart Weight) and the Y variable (Body Weight). 
"""