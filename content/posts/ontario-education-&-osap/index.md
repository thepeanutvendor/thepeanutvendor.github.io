---
title: "Ontario Education & OSAP"
date: 2026-04-13
draft: false
description: "In this notebook, I quantify returns to education, evaluate them in view of OSAP financing, and also analyse ROI in terms of field of study, gender, and other variables."
series: "Empirical Economics"
series_order: 2
---

# Ontario Education & OSAP

This notebook aims to quantify returns to post-secondary education (covered under OSAP), analyse OSAP's latest policies, and see if the one matches the other - if getting tertiary education is worth it given OSAP financing arrangements, especially loan burdens.

For this notebook, I will be using [StatCan's Census 2021 PUMF Individual data](https://www150.statcan.gc.ca/n1/pub/98m0001x/index-eng.htm). Since we don't want to analyse structures (i.e. families), we don't need hierarchical data.

To quantify and analyse the returns to post-secondary education, we need a few things. First, we need a metric to compare against - the median income of someone who won't get post-secondary education. A counterfactual would be ideal - the sort of person who gets post-secondary education is not guaranteed to earn the same in, say, the trades as a tradesman. They may earn more, or they may earn less. Instead of comparing Person A who gets post-secondary education to Person B who doesn't, a detailed analysis would compare Person A who gets post-secondary education to Person A who doesn't. But constructing such a model seems complex, so we will simply compare median earnings.

Once we've constructed that model, we need to quantify the returns to post-secondary education. Median wages vary by degree, field, and experience, even gender, so we will need to control or analyse these variables separately.

Finally, we'll need to do some accounting calculations: given OSAP's financing burden, does an investment in education (given the above models) make sense? In technical terms, I'm going to calculate the NPV (value of investment) of the earnings premium (of education) net (minus) of loan repayment costs, and maybe compared across grant/loan ratios.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
```

## Pre-processing

Census 2021's PUMF data is massive. I will filter the file prior to working with it to minimise space-used. It is also complex, primarily using codes for characteristics and values. I will handle this all early.

The characteristics I'm filtering for:
1. PPSORT - Unique ID
2. AGEGRP - AGE
3. CIP2021 - Fields of Study
4. EmpIn - Income: Employment income
5. Gender - Gender
6. HDGREE - Education: Highest certificate, diploma or degree
7. LFACT - Labour: Labour force status - Detailed
8. PR - Province


```python
# Filter for variables and load csv
cols = ['PPSORT', 'AGEGRP', 'CIP2021', 'EmpIn', 'Gender', 'SSGRAD', 'LFACT', 'PR']
census_2021 = pd.read_csv('census_2021_ontario.csv', usecols=cols)
# Filter for Ontario only
census_2021 = census_2021[census_2021['PR'] == 35]
# Remove NA
census_2021 = census_2021[census_2021['EmpIn'] != 99999999]
census_2021 = census_2021[census_2021['EmpIn'] != 88888888]
census_2021 = census_2021[~census_2021['SSGRAD'].isin([88, 99])]
census_2021 = census_2021[~census_2021['AGEGRP'].isin([88])]
census_2021 = census_2021[~census_2021['CIP2021'].isin([88, 99])]
census_2021 = census_2021[~census_2021['LFACT'].isin([88, 99])]
# Filtering for working age
census_2021 = census_2021[census_2021['AGEGRP'].isin(range(8,17))]
# Filter for positive income
census_2021 = census_2021[census_2021['EmpIn'] >= 0]

census_2021.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PPSORT</th>
      <th>AGEGRP</th>
      <th>CIP2021</th>
      <th>EmpIn</th>
      <th>Gender</th>
      <th>LFACT</th>
      <th>PR</th>
      <th>SSGRAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>11</td>
      <td>8</td>
      <td>12000</td>
      <td>1</td>
      <td>3</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>16</td>
      <td>4</td>
      <td>61000</td>
      <td>1</td>
      <td>13</td>
      <td>35</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>12</td>
      <td>13</td>
      <td>25000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>13</td>
      <td>5</td>
      <td>130000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21</td>
      <td>10</td>
      <td>5</td>
      <td>63000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
SSGRAD = {
    1:  'No certificate',
    2:  'Trades (no HS)',
    3:  'College/CEGEP (no HS)',
    4:  'HS',
    5:  'Trades',
    6:  'College/CEGEP',
    7:  'University below bachelor',
    8:  'Bachelor',
    9:  'University above bachelor',
    10: 'Medicine/Dentistry/Vet/Optometry',
    11: 'Masters',
    12: 'Doctorate',
    88: None,  # Not available
    99: None   # Not applicable
}


GENDER = {
    1: "Woman",
    2: "Man"
}

AGEGRP = {
    4: "10-11",
    19: "75-79",
    16: "60-64",
    10: "30-34",
    11: "35-39",
    2: "5-6",
    17: "65-69",
    3: "7-9",
    1: "0-4",
    20: "80-84",
    12: "40-44",
    8: "20-24",
    6: "15-17",
    7: "18-19",
    18: "70-74",
    13: "45-49",
    88: None,
    14: "50-54",
    9: "25-29",
    21: "85+",
    5: "12-14",
    15: "55-59",
}

CIP2021 = {
    11: "Personal/protective/transport services",
    8: "Architecture/engineering/trades",
    99: None,
    3: "Humanities",
    10: "Health",
    7: "Math/CS/info",
    88: None,
    4: "Social sciences & law",
    9: "Agriculture/natural resources/conservation",
    5: "Business/management/public admin",
    1: "Education",
    13: "No postsecondary degree",
    12: "Other",
    2: "Visual/performing arts & comm",
    6: "Physical/life sciences & tech",
}

LFACT = {
    1:  'Employed',
    2:  'Employed - Absent',
    3:  'Unemployed',
    4:  'Unemployed',
    5:  'Unemployed',
    6:  'Unemployed',
    7:  'Unemployed',
    8:  'Unemployed',
    9:  'Unemployed',
    10: 'Unemployed',
    11: 'Not in labour force',
    12: 'Not in labour force',
    13: 'Not in labour force',
    14: 'Not in labour force',
    88: None,
    99: None
}
```

## Constructing the Alternate

First, we will find the median income of the average non-post secondary educated person. This includes those with a high school education, tradesmen, career college diplomas, and so on.


```python
non_ps_df = census_2021[census_2021['SSGRAD'].isin([1, 2, 4, 5])]
non_ps_df['EmpIn'].median()
```




    np.float64(33000.0)



The median non-post secondary educated person in Ontario earns $33,000 per year. This figure accounts for the unemployed, and not for those with negative income.


```python
non_ps_df[non_ps_df['LFACT'].isin([1, 2])]['EmpIn'].median()
```




    np.float64(42000.0)



Filtering for the employed only, that number increases to $42000 per year.

## Returns to Education

The Mincer earnings function, given wages (their log, to be specific), years of experience, and years of education (alongside other independent variables), models an individuals earnings as a function of these variables, allowing us to calculate the value of, for instance, an additional year of schooling. It's identical to what I did in the last post, with two distinctions - first, we're tracking multiple independent variables (schooling, exp, gender); second, we're using the log of wages. This is because we want the percentage increase of our coefficients, not their absolute increase.

There are some limitations, specific to my data and general to the model. I only have age groups, so I will have to use midpoints - this affects years of experience. Furthermore, my formula for years of experience is simply: age - (years of schooling + 6). This does not account for unemployment and other such things.

Years of education are guessed for some groups with non-post-secondary education - it is impossible to get an exact year for, say, someone who has neither an HS diploma or degree.

### Code


```python
mincer_df = census_2021.copy()

# Map degree to years of schooling
edu_years = {
    1:  10,  # No certificate — assume some high school
    2:  11,  # Trades no HS — HS dropout + trades
    3:  13,  # College no HS — assume completed college
    4:  12,  # HS diploma only
    5:  13,  # Trades + HS — 12 + ~1 year trades
    6:  14,  # College/CEGEP — 12 + 2
    7:  14,  # University below bachelor — 12 + 2
    8:  16,  # Bachelor — 12 + 4
    9:  17,  # University above bachelor — 12 + 5
    10: 18,  # Medicine/Dentistry/Vet — 12 + 4 + 2 specialty minimum
    11: 18,  # Masters — 12 + 4 + 2
    12: 21,  # Doctorate — 12 + 4 + 5
}
age_midpoint = {
    8:  22,   # 20-24
    9:  27,   # 25-29
    10: 32,   # 30-34
    11: 37,   # 35-39
    12: 42,   # 40-44
    13: 47,   # 45-49
    14: 52,   # 50-54
    15: 57,   # 55-59
    16: 62,   # 60-64
}

mincer_df['log_wage'] = np.log(mincer_df['EmpIn'])
mincer_df['age_mid'] = mincer_df['AGEGRP'].map(age_midpoint)
mincer_df['edu_years'] = mincer_df['SSGRAD'].map(edu_years)
mincer_df['exp_years'] = mincer_df['age_mid'] - (mincer_df['edu_years'] + 6)
mincer_df['exp_years'] = mincer_df['exp_years'].clip(lower=0)

mincer_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PPSORT</th>
      <th>AGEGRP</th>
      <th>CIP2021</th>
      <th>EmpIn</th>
      <th>Gender</th>
      <th>LFACT</th>
      <th>PR</th>
      <th>SSGRAD</th>
      <th>log_wage</th>
      <th>age_mid</th>
      <th>edu_years</th>
      <th>exp_years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>11</td>
      <td>8</td>
      <td>12000</td>
      <td>1</td>
      <td>3</td>
      <td>35</td>
      <td>6</td>
      <td>9.392662</td>
      <td>37</td>
      <td>14</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>16</td>
      <td>4</td>
      <td>61000</td>
      <td>1</td>
      <td>13</td>
      <td>35</td>
      <td>11</td>
      <td>11.018629</td>
      <td>62</td>
      <td>18</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>12</td>
      <td>13</td>
      <td>25000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>4</td>
      <td>10.126631</td>
      <td>42</td>
      <td>12</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>13</td>
      <td>5</td>
      <td>130000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>8</td>
      <td>11.775290</td>
      <td>47</td>
      <td>16</td>
      <td>25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21</td>
      <td>10</td>
      <td>5</td>
      <td>63000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>11</td>
      <td>11.050890</td>
      <td>32</td>
      <td>18</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>378842</th>
      <td>980851</td>
      <td>11</td>
      <td>8</td>
      <td>180000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>12</td>
      <td>12.100712</td>
      <td>37</td>
      <td>21</td>
      <td>10</td>
    </tr>
    <tr>
      <th>378843</th>
      <td>980852</td>
      <td>12</td>
      <td>13</td>
      <td>49000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>10.799576</td>
      <td>42</td>
      <td>10</td>
      <td>26</td>
    </tr>
    <tr>
      <th>378844</th>
      <td>980856</td>
      <td>12</td>
      <td>10</td>
      <td>110000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>6</td>
      <td>11.608236</td>
      <td>42</td>
      <td>14</td>
      <td>22</td>
    </tr>
    <tr>
      <th>378846</th>
      <td>980862</td>
      <td>16</td>
      <td>13</td>
      <td>33000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>4</td>
      <td>10.404263</td>
      <td>62</td>
      <td>12</td>
      <td>44</td>
    </tr>
    <tr>
      <th>378848</th>
      <td>980866</td>
      <td>10</td>
      <td>5</td>
      <td>130000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>6</td>
      <td>11.775290</td>
      <td>32</td>
      <td>14</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>182864 rows × 12 columns</p>
</div>




```python
model = sm.OLS.from_formula('log_wage ~ edu_years + exp_years + I(exp_years**2) + C(Gender)', data=mincer_df)
results = model.fit()

results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>log_wage</td>     <th>  R-squared:         </th>  <td>   0.102</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.102</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   5217.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 13 Apr 2026</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:07:21</td>     <th>  Log-Likelihood:    </th> <td>-3.3964e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>182864</td>      <th>  AIC:               </th>  <td>6.793e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>182859</td>      <th>  BIC:               </th>  <td>6.793e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    7.0040</td> <td>    0.027</td> <td>  264.176</td> <td> 0.000</td> <td>    6.952</td> <td>    7.056</td>
</tr>
<tr>
  <th>C(Gender)[T.2]</th>    <td>    0.3884</td> <td>    0.007</td> <td>   53.291</td> <td> 0.000</td> <td>    0.374</td> <td>    0.403</td>
</tr>
<tr>
  <th>edu_years</th>         <td>    0.1497</td> <td>    0.002</td> <td>   90.663</td> <td> 0.000</td> <td>    0.146</td> <td>    0.153</td>
</tr>
<tr>
  <th>exp_years</th>         <td>    0.1044</td> <td>    0.001</td> <td>   94.200</td> <td> 0.000</td> <td>    0.102</td> <td>    0.107</td>
</tr>
<tr>
  <th>I(exp_years ** 2)</th> <td>   -0.0020</td> <td> 2.43e-05</td> <td>  -80.547</td> <td> 0.000</td> <td>   -0.002</td> <td>   -0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>142020.638</td> <th>  Durbin-Watson:     </th>  <td>   1.997</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>3451730.600</td>
</tr>
<tr>
  <th>Skew:</th>            <td>-3.609</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>        <td>23.023</td>   <th>  Cond. No.          </th>  <td>6.39e+03</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.39e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Since we are using `ln(wage)` (we have a log-level model), we need to adjust the coefficients to get proper slopes.


```python
params = results.params
adjusted_params = (np.exp(params) - 1) * 100
adjusted_params
```




    Intercept            110006.516586
    C(Gender)[T.2]           47.455093
    edu_years                16.143483
    exp_years                10.999062
    I(exp_years ** 2)        -0.195553
    dtype: float64



### Interpretation

We have three independent variables we're tracking: years of schooling, experience, and gender. Our `R-squared` is equal to `0.102`, which means that 10% of the variation in employment income is explained by these factors.  All these variables are statistically significant (`P>|t| < 0.05`).

Our baseline is a male with 0 years of education and experience. An additional year of schooling (for such a person) gives us a `16.1%` increase in wages. An additional year of experience gives us `11%` more wages, but with a diminishing return of `0.2pp` each year.

Our data also shows that men earn `47.46%` more than women.

Importantly, we can now model incomes based on a hypothetical person's education, experience, and gender.


```python
baseline = results.predict({'edu_years': 12, 'exp_years': 2, 'Gender': 2})
# HS degree, 2 years of experience, male
np.exp(baseline[0])
```




    np.float64(11957.7402224044)



## The Costs of Education

The purpose of this article is to see how OSAP policy changes affect an individual's choice of education. Starting from the year 2026-27, OSAP has tweaked its grant:loans ratio to 25:75, at a maximum. The exact ratio may be 0:100, depending on an individual's financial background. We will be testing variations up to the maximum.

One question is how *much* money our model should expect from OSAP. Again, this depends on an individual's background. Some individuals can expect a near 100% cover, while others will receive nothing. Moreover, the ratio and level of financing are tightly linked: an individual who receives a 100% cover will likely also get the maximum 25:75 ratio because both factors are determined by financial background.

Generally, the costs of post-secondary education are many. Our model covers experience - every year spent in post-secondary is one less year of experience compared to the person who skipped it. Some costs are deceptive - we could include rent and other living expenses, but these are independent of post-secondary education.

## Simple NPV of Education

To cut through the fog, we will start with a simple calculation. We will ignore OSAP and other factors, and simply see if, ceteris paribus, education is a worthwhile investment using the Net Present Value formula. The NPV formula is expressed as such:

`NPV = sum(net_cash_flow / (1 + discount_rate)^time) - costs`

Our costs will be the costs of education (tuition, fees) and the foregone wages (opportunity cost). [Ontariocolleges.ca](https://www.ontariocolleges.ca/en/fees-and-aid/tuition) provides a number: 6100 (tuition) + 800 (fees) + 1300 (books and supplies for a total $8200 per academic year. I will assume a bachelors degree. Note that this is an average, exact costs will depend on institution. It also assumes Ontario, despite OSAP covering out-of-province study in certain cases. Our discount rate will be the rate of interest. Our net cash flow will be earnings post-degree. Time will be a working lifetime. Gender will be male (for our example).

This can be quite confusing. It serves to express this in tabular form once to get the general gist of how it works. First, we need to get our cash flows down.


```python
post_sec = []
post_sec[0:3] = [-8200] * 4 # adding college costs
# getting the rest of the cash flow of our post-sec model
for exp in range(22,66):
    post_sec.append(np.exp(results.predict({'edu_years': 16, 'exp_years': exp-22, 'Gender': 2})[0]))
hs = [] # same but for our hs model
for exp in range(18,66):
    hs.append(np.exp(results.predict({'edu_years': 12, 'exp_years': exp-18, 'Gender': 2})[0]))
data = [post_sec, hs]
cash_flow = pd.DataFrame(data=list(zip(*data)), columns=['post_sec', 'hs'], index=range(0, 48))
cash_flow
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_sec</th>
      <th>hs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-8200.000000</td>
      <td>9781.623220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-8200.000000</td>
      <td>10836.277815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-8200.000000</td>
      <td>11957.740222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-8200.000000</td>
      <td>13143.707352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17798.782943</td>
      <td>14390.849470</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19717.847682</td>
      <td>15694.763037</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21758.476883</td>
      <td>17049.940925</td>
    </tr>
    <tr>
      <th>7</th>
      <td>23916.479808</td>
      <td>18449.762380</td>
    </tr>
    <tr>
      <th>8</th>
      <td>26185.797625</td>
      <td>19886.504769</td>
    </tr>
    <tr>
      <th>9</th>
      <td>28558.417591</td>
      <td>21351.378687</td>
    </tr>
    <tr>
      <th>10</th>
      <td>31024.318856</td>
      <td>22834.587510</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33571.454203</td>
      <td>24325.411815</td>
    </tr>
    <tr>
      <th>12</th>
      <td>36185.771412</td>
      <td>25812.318479</td>
    </tr>
    <tr>
      <th>13</th>
      <td>38851.277159</td>
      <td>27283.093505</td>
    </tr>
    <tr>
      <th>14</th>
      <td>41550.145363</td>
      <td>28724.996933</td>
    </tr>
    <tr>
      <th>15</th>
      <td>44262.870809</td>
      <td>30124.937429</td>
    </tr>
    <tr>
      <th>16</th>
      <td>46968.467659</td>
      <td>31469.663498</td>
    </tr>
    <tr>
      <th>17</th>
      <td>49644.711148</td>
      <td>32745.967587</td>
    </tr>
    <tr>
      <th>18</th>
      <td>52268.419458</td>
      <td>33940.898824</td>
    </tr>
    <tr>
      <th>19</th>
      <td>54815.771412</td>
      <td>35041.979698</td>
    </tr>
    <tr>
      <th>20</th>
      <td>57262.654397</td>
      <td>36037.421651</td>
    </tr>
    <tr>
      <th>21</th>
      <td>59585.035758</td>
      <td>36916.334435</td>
    </tr>
    <tr>
      <th>22</th>
      <td>61759.349903</td>
      <td>37668.924029</td>
    </tr>
    <tr>
      <th>23</th>
      <td>63762.892569</td>
      <td>38286.674120</td>
    </tr>
    <tr>
      <th>24</th>
      <td>65574.213131</td>
      <td>38762.506430</td>
    </tr>
    <tr>
      <th>25</th>
      <td>67173.495531</td>
      <td>39090.915696</td>
    </tr>
    <tr>
      <th>26</th>
      <td>68542.918431</td>
      <td>39268.075679</td>
    </tr>
    <tr>
      <th>27</th>
      <td>69666.985421</td>
      <td>39291.913367</td>
    </tr>
    <tr>
      <th>28</th>
      <td>70532.816768</td>
      <td>39162.149351</td>
    </tr>
    <tr>
      <th>29</th>
      <td>71130.395013</td>
      <td>38880.303273</td>
    </tr>
    <tr>
      <th>30</th>
      <td>71452.757879</td>
      <td>38449.664208</td>
    </tr>
    <tr>
      <th>31</th>
      <td>71496.133281</td>
      <td>37875.226776</td>
    </tr>
    <tr>
      <th>32</th>
      <td>71260.012801</td>
      <td>37163.594721</td>
    </tr>
    <tr>
      <th>33</th>
      <td>70747.161605</td>
      <td>36322.854566</td>
    </tr>
    <tr>
      <th>34</th>
      <td>69963.564541</td>
      <td>35362.422716</td>
    </tr>
    <tr>
      <th>35</th>
      <td>68918.309890</td>
      <td>34292.870055</td>
    </tr>
    <tr>
      <th>36</th>
      <td>67623.413921</td>
      <td>33125.728596</td>
    </tr>
    <tr>
      <th>37</th>
      <td>66093.590987</td>
      <td>31873.285126</td>
    </tr>
    <tr>
      <th>38</th>
      <td>64345.975315</td>
      <td>30548.366986</td>
    </tr>
    <tr>
      <th>39</th>
      <td>62399.801840</td>
      <td>29164.125185</td>
    </tr>
    <tr>
      <th>40</th>
      <td>60276.054377</td>
      <td>27733.819916</td>
    </tr>
    <tr>
      <th>41</th>
      <td>57997.090143</td>
      <td>26270.613299</td>
    </tr>
    <tr>
      <th>42</th>
      <td>55586.249953</td>
      <td>24787.373747</td>
    </tr>
    <tr>
      <th>43</th>
      <td>53067.463571</td>
      <td>23296.495863</td>
    </tr>
    <tr>
      <th>44</th>
      <td>50464.859437</td>
      <td>21809.739138</td>
    </tr>
    <tr>
      <th>45</th>
      <td>47802.387535</td>
      <td>20338.088067</td>
    </tr>
    <tr>
      <th>46</th>
      <td>45103.463416</td>
      <td>18891.635562</td>
    </tr>
    <tr>
      <th>47</th>
      <td>42390.640477</td>
      <td>17479.490823</td>
    </tr>
  </tbody>
</table>
</div>




```python
# crunching the numbers
cash_flow['net_ps'] = cash_flow['post_sec'] / (1 + 0.03) ** cash_flow.index
cash_flow['net_hs']= cash_flow['hs'] / (1 + 0.03) ** cash_flow.index
male_npv = cash_flow['net_ps'].sum() - cash_flow['net_hs'].sum()
male_npv
```




    np.float64(356221.5858905276)



The data seems to tell us that college education is a great investment. A NPV of $356,222, specifically.

There's a few complications here, though. Of course, financing the initial investment is difficult, which is why we have to bring OSAP into our calculations. This doesn't present the whole story.

The main specific issue is unemployment - my numbers account for unemployment, so the unemployed are dragging the wages down (hence the somewhat odd, below minimum wage, numbers for initial years). Since we're doing this for both HS and PS models, it cancels out somewhat. Employment chances are pretty important for decisions such as these, so accounting for unemployment seems sensible, but it does make our figures somewhat less precise.

Moreover, we're assuming a Bachelor's degree, and not accounting for field of study. For our non-post-sec model, we're only assuming a high-school degree when trades are also an option. These variables will be covered later on.

For clarity's sake, I'll recrunch the numbers for a female model too.


```python
post_sec_fem = []
post_sec_fem[0:3] = [-8200] * 4
for exp in range(22,66):
    post_sec_fem.append(np.exp(results.predict({'edu_years': 16, 'exp_years': exp-22, 'Gender': 1})[0]))
hs_fem = []
for exp in range(18,66):
    hs_fem.append(np.exp(results.predict({'edu_years': 12, 'exp_years': exp-18, 'Gender': 1})[0]))
data = [post_sec_fem, hs_fem]
cash_flow_fem = pd.DataFrame(data=list(zip(*data)), columns=['post_sec', 'hs'], index=range(0, 48))
cash_flow_fem
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_sec</th>
      <th>hs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-8200.000000</td>
      <td>6633.628607</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-8200.000000</td>
      <td>7348.866430</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-8200.000000</td>
      <td>8109.411478</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-8200.000000</td>
      <td>8913.701861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12070.646461</td>
      <td>9759.479442</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13372.103537</td>
      <td>10643.757863</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14756.002297</td>
      <td>11562.802341</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16219.500697</td>
      <td>12512.122862</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17758.489805</td>
      <td>13486.482148</td>
    </tr>
    <tr>
      <th>9</th>
      <td>19367.535597</td>
      <td>14479.919465</td>
    </tr>
    <tr>
      <th>10</th>
      <td>21039.842208</td>
      <td>15485.791012</td>
    </tr>
    <tr>
      <th>11</th>
      <td>22767.239544</td>
      <td>16496.827169</td>
    </tr>
    <tr>
      <th>12</th>
      <td>24540.197778</td>
      <td>17505.206490</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26347.870675</td>
      <td>18502.645778</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28178.169074</td>
      <td>19480.505138</td>
    </tr>
    <tr>
      <th>15</th>
      <td>30017.865075</td>
      <td>20429.906389</td>
    </tr>
    <tr>
      <th>16</th>
      <td>31852.726658</td>
      <td>21341.862730</td>
    </tr>
    <tr>
      <th>17</th>
      <td>33667.681597</td>
      <td>22207.417160</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35447.008618</td>
      <td>23017.786755</td>
    </tr>
    <tr>
      <th>19</th>
      <td>37174.552852</td>
      <td>23764.509606</td>
    </tr>
    <tr>
      <th>20</th>
      <td>38833.961786</td>
      <td>24439.591039</td>
    </tr>
    <tr>
      <th>21</th>
      <td>40408.937134</td>
      <td>25035.645585</td>
    </tr>
    <tr>
      <th>22</th>
      <td>41883.497357</td>
      <td>25546.031208</td>
    </tr>
    <tr>
      <th>23</th>
      <td>43242.245046</td>
      <td>25964.972378</td>
    </tr>
    <tr>
      <th>24</th>
      <td>44470.632975</td>
      <td>26287.668801</td>
    </tr>
    <tr>
      <th>25</th>
      <td>45555.222439</td>
      <td>26510.386958</td>
    </tr>
    <tr>
      <th>26</th>
      <td>46483.927494</td>
      <td>26630.532000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>47246.238899</td>
      <td>26646.698067</td>
    </tr>
    <tr>
      <th>28</th>
      <td>47833.421973</td>
      <td>26558.695669</td>
    </tr>
    <tr>
      <th>29</th>
      <td>48238.683150</td>
      <td>26367.555389</td>
    </tr>
    <tr>
      <th>30</th>
      <td>48457.300804</td>
      <td>26075.507786</td>
    </tr>
    <tr>
      <th>31</th>
      <td>48486.716813</td>
      <td>25685.940073</td>
    </tr>
    <tr>
      <th>32</th>
      <td>48326.586379</td>
      <td>25203.330730</td>
    </tr>
    <tr>
      <th>33</th>
      <td>47978.784763</td>
      <td>24633.163814</td>
    </tr>
    <tr>
      <th>34</th>
      <td>47447.370724</td>
      <td>23981.825273</td>
    </tr>
    <tr>
      <th>35</th>
      <td>46738.507686</td>
      <td>23256.483991</td>
    </tr>
    <tr>
      <th>36</th>
      <td>45860.344752</td>
      <td>22464.960662</td>
    </tr>
    <tr>
      <th>37</th>
      <td>44822.860793</td>
      <td>21615.587849</td>
    </tr>
    <tr>
      <th>38</th>
      <td>43637.675773</td>
      <td>20717.064702</td>
    </tr>
    <tr>
      <th>39</th>
      <td>42317.834296</td>
      <td>19778.309875</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40877.567011</td>
      <td>18808.316068</td>
    </tr>
    <tr>
      <th>41</th>
      <td>39332.035967</td>
      <td>17816.009469</td>
    </tr>
    <tr>
      <th>42</th>
      <td>37697.070267</td>
      <td>16810.117083</td>
    </tr>
    <tr>
      <th>43</th>
      <td>35988.898421</td>
      <td>15799.044589</td>
    </tr>
    <tr>
      <th>44</th>
      <td>34223.883674</td>
      <td>14790.766952</td>
    </tr>
    <tr>
      <th>45</th>
      <td>32418.268248</td>
      <td>13792.733556</td>
    </tr>
    <tr>
      <th>46</th>
      <td>30587.931928</td>
      <td>12811.789136</td>
    </tr>
    <tr>
      <th>47</th>
      <td>28748.169810</td>
      <td>11854.111302</td>
    </tr>
  </tbody>
</table>
</div>




```python
# crunching the numbers
cash_flow_fem['net_ps'] = cash_flow_fem['post_sec'] / (1 + 0.03) ** cash_flow_fem.index
cash_flow_fem['net_hs']= cash_flow_fem['hs'] / (1 + 0.03) ** cash_flow_fem.index
fem_npv = cash_flow_fem['net_ps'].sum() - cash_flow_fem['net_hs'].sum()
fem_npv
```




    np.float64(231476.0626947455)



For women, the NPV is $231,476. This is an interesting result - common-sense tells us that women benefit more from college than men because the sort of jobs that don't require college degrees are less-suited for them (for whatever reason). This value seems to contradict that.

However, this is an absolute number. It makes sense for the absolute number to be lower, because of the gender pay gap we quantified earlier in this notebook. Once we account for that, by modeling the relative ROI, what do the results look like?


```python
hs_male_pv = cash_flow['net_hs'].sum()
hs_fem_pv = cash_flow_fem['net_hs'].sum()

male_roi = (male_npv / hs_male_pv) * 100
fem_roi = (fem_npv / hs_fem_pv) * 100

print(f"Male ROI: {male_roi:.1f}%")
print(f"Female ROI: {fem_roi:.1f}%")
```

    Male ROI: 52.2%
    Female ROI: 50.0%


The results seem to be similar. This does not contradict the intuition, however. Our model fails to capture some considerations. For starters, our coefficient for edu_years already accounts for both female *and* male returns together. We'd have to calculate two seperate slopes, and *then* our result would be cleaner. The exact idea behind women and edcation is quite complex, and goes beyond just a simple wage model's capabilities too. That's for another day, though.

## Accounting for OSAP

Regardless, now that we have a base model, and have determined that education makes sense *if* you can afford initial costs, we can move onto accounting for OSAP. We will assume a 100% OSAP coverage, with variations in grants:loans ratio. Why? Well, if you have 0% OSAP coverage you can either not afford college or will pay using a RESP or some other means: for the former, this model is useless in any case, and in the latter case the answer is yes - our 'naive' model says you should go to college.

The case of varying proportions of OSAP and misc coverage is more complex (i.e. 50% OSAP, 50% other sources). However, there is no need to cover that. This is because, if 100% OSAP (meaning a large loan burden) is still a positive investment, then a lower proportion of OSAP funds should generally be the same as well.

OSAP, including loans, does cover non-tuition/fees/books expenses. I made the decision not to include these since these exist for HS too, and I won't include them in this model either.

OSAP loans need to be repayed 6 months after graduating (for simplicities' sake, I'll put that as starting in the fifth year). The time horizon is 9.5 years. 70% of the debt is 0-interest federal, while 30% is provincial and has an interest rate of: `prime-rate + 1%`. The current prime-rate is 4.45%, but this varies. OSAP has a [handy calculator](https://osap.gov.on.ca/AidEstimator2526Web/enterapp/debt_calculator.xhtml) that contains more details.

Once we are done with that, we will finally control for fields of study and exact degree variation. 


```python
def osap_loan(grant_ratio=0.25, total_osap=8200*4, interest_rate=0.06):
    """Given grant_ratio, loan, and interest rate, calculates and returns annuity"""
    loan = total_osap * (1 - grant_ratio)
    loan_fed = loan * 0.75
    loan_prov = loan * 0.25

    annual_payment = loan_prov * (interest_rate * (1 + interest_rate)**10) / ((1 + interest_rate)**10 - 1)
    annual_payment += loan_fed / 10

    return annual_payment

osap_loan()
```




    2680.58794305536




```python
def osap_npv(ratio, main_edu=16, sec_edu=12, main_model=results, sec_model=results, bFos=False, fos=None):
    """Given a OLS model, education for main and counterfactual model, grant ratio, returns NPV for each gender"""
    main_male = [0 for x in range(18,66)]
    sec_male = [0 for x in range(18,66)]
    main_female = [0 for x in range(18,66)]
    sec_female = [0 for x in range(18,66)]
    sec_range = sec_edu - 12

    for exp in range(22,66):
        if bFos:
            main_male[exp - 18] = np.exp(main_model.predict({'edu_years': main_edu, 'exp_years': exp-22, 'Gender': 2, 'FOS': fos})[0])
            main_female[exp - 18] = np.exp(main_model.predict({'edu_years': main_edu, 'exp_years': exp-22, 'Gender': 1, 'FOS': fos})[0])
        else:
            main_male[exp - 18] = np.exp(main_model.predict({'edu_years': main_edu, 'exp_years': exp-22, 'Gender': 2})[0])
            main_female[exp - 18] = np.exp(main_model.predict({'edu_years': main_edu, 'exp_years': exp-22, 'Gender': 1})[0])

    for exp in range(18 + sec_range,66):
        if bFos:
            sec_male[exp - 18] = np.exp(sec_model.predict({'edu_years': sec_edu, 'exp_years': exp-(18 + sec_range), 'Gender': 2, 'FOS': fos})[0])
            sec_female[exp - 18] = np.exp(sec_model.predict({'edu_years': sec_edu, 'exp_years': exp-(18 + sec_range), 'Gender': 1, 'FOS': fos})[0])
        else:
            sec_male[exp - 18] = np.exp(sec_model.predict({'edu_years': sec_edu, 'exp_years': exp-(18 + sec_range), 'Gender': 2})[0])
            sec_female[exp - 18] = np.exp(sec_model.predict({'edu_years': sec_edu, 'exp_years': exp-(18 + sec_range), 'Gender': 1})[0])

    annual_payment = osap_loan(grant_ratio=ratio/100)
        
    main_male[4:14] = [x - annual_payment for x in main_male[4:14]]
    main_female[4:14] = [x - annual_payment for x in main_female[4:14]]

    data = [main_male, sec_male, main_female, sec_female]
    cash_flow_osap = pd.DataFrame(data=list(zip(*data)), columns=['main_male', 'sec_male', 'main_female', 'sec_female'], index=range(0, 48))
    cash_flow_osap['net_main_male'] = cash_flow_osap['main_male'] / (1 + 0.03) ** cash_flow_osap.index
    cash_flow_osap['net_sec_male']= cash_flow_osap['sec_male'] / (1 + 0.03) ** cash_flow_osap.index
    cash_flow_osap['net_main_female'] = cash_flow_osap['main_female'] / (1 + 0.03) ** cash_flow_osap.index
    cash_flow_osap['net_sec_female']= cash_flow_osap['sec_female'] / (1 + 0.03) ** cash_flow_osap.index
    return[cash_flow_osap['net_main_male'].sum() - cash_flow_osap['net_sec_male'].sum(), cash_flow_osap['net_main_female'].sum() - cash_flow_osap['net_sec_female'].sum() ]
```


```python
ratios = range(0,26) # from 0% grants to 25% grants, the maximum
osap_npvs = []
for ratio in ratios:
    osap_npvs.append(osap_npv(ratio=ratio))

pd.DataFrame(osap_npvs, columns=['Male', 'Female'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
      <th>Female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>359715.410269</td>
      <td>234969.887073</td>
    </tr>
    <tr>
      <th>1</th>
      <td>359994.418157</td>
      <td>235248.894961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>360273.426044</td>
      <td>235527.902848</td>
    </tr>
    <tr>
      <th>3</th>
      <td>360552.433931</td>
      <td>235806.910735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>360831.441819</td>
      <td>236085.918623</td>
    </tr>
    <tr>
      <th>5</th>
      <td>361110.449706</td>
      <td>236364.926510</td>
    </tr>
    <tr>
      <th>6</th>
      <td>361389.457593</td>
      <td>236643.934397</td>
    </tr>
    <tr>
      <th>7</th>
      <td>361668.465480</td>
      <td>236922.942285</td>
    </tr>
    <tr>
      <th>8</th>
      <td>361947.473368</td>
      <td>237201.950172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>362226.481255</td>
      <td>237480.958059</td>
    </tr>
    <tr>
      <th>10</th>
      <td>362505.489142</td>
      <td>237759.965947</td>
    </tr>
    <tr>
      <th>11</th>
      <td>362784.497030</td>
      <td>238038.973834</td>
    </tr>
    <tr>
      <th>12</th>
      <td>363063.504917</td>
      <td>238317.981721</td>
    </tr>
    <tr>
      <th>13</th>
      <td>363342.512804</td>
      <td>238596.989609</td>
    </tr>
    <tr>
      <th>14</th>
      <td>363621.520692</td>
      <td>238875.997496</td>
    </tr>
    <tr>
      <th>15</th>
      <td>363900.528579</td>
      <td>239155.005383</td>
    </tr>
    <tr>
      <th>16</th>
      <td>364179.536466</td>
      <td>239434.013271</td>
    </tr>
    <tr>
      <th>17</th>
      <td>364458.544354</td>
      <td>239713.021158</td>
    </tr>
    <tr>
      <th>18</th>
      <td>364737.552241</td>
      <td>239992.029045</td>
    </tr>
    <tr>
      <th>19</th>
      <td>365016.560128</td>
      <td>240271.036932</td>
    </tr>
    <tr>
      <th>20</th>
      <td>365295.568016</td>
      <td>240550.044820</td>
    </tr>
    <tr>
      <th>21</th>
      <td>365574.575903</td>
      <td>240829.052707</td>
    </tr>
    <tr>
      <th>22</th>
      <td>365853.583790</td>
      <td>241108.060594</td>
    </tr>
    <tr>
      <th>23</th>
      <td>366132.591677</td>
      <td>241387.068482</td>
    </tr>
    <tr>
      <th>24</th>
      <td>366411.599565</td>
      <td>241666.076369</td>
    </tr>
    <tr>
      <th>25</th>
      <td>366690.607452</td>
      <td>241945.084256</td>
    </tr>
  </tbody>
</table>
</div>



The results are about as clear as they get - post-secondary education, no matter the grant ratio, is a good investment *relative* to going through life with a high school diploma. The main reasons for the 'no matter' part are simple: OSAP loans are 70% federal, which is zero-interest. Zero-interest loans are no different from the one-time expenses we considered in the naive model. Of course, since up to 25% of your allowance is grants - free money - that helps out too.

Of course, most critics of college do not recommend just a high school diploma. They often recommend the skilled trades.


```python
ratios = range(0,26) # from 0% grants to 25% grants, the maximum
osap_npvs = []
for ratio in ratios:
    osap_npvs.append(osap_npv(sec_edu=13, ratio=ratio))

pd.DataFrame(osap_npvs, columns=['Male', 'Female'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
      <th>Female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>277557.536966</td>
      <td>179252.670575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>277836.544853</td>
      <td>179531.678463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>278115.552741</td>
      <td>179810.686350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>278394.560628</td>
      <td>180089.694237</td>
    </tr>
    <tr>
      <th>4</th>
      <td>278673.568515</td>
      <td>180368.702125</td>
    </tr>
    <tr>
      <th>5</th>
      <td>278952.576403</td>
      <td>180647.710012</td>
    </tr>
    <tr>
      <th>6</th>
      <td>279231.584290</td>
      <td>180926.717899</td>
    </tr>
    <tr>
      <th>7</th>
      <td>279510.592177</td>
      <td>181205.725787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>279789.600065</td>
      <td>181484.733674</td>
    </tr>
    <tr>
      <th>9</th>
      <td>280068.607952</td>
      <td>181763.741561</td>
    </tr>
    <tr>
      <th>10</th>
      <td>280347.615839</td>
      <td>182042.749449</td>
    </tr>
    <tr>
      <th>11</th>
      <td>280626.623727</td>
      <td>182321.757336</td>
    </tr>
    <tr>
      <th>12</th>
      <td>280905.631614</td>
      <td>182600.765223</td>
    </tr>
    <tr>
      <th>13</th>
      <td>281184.639501</td>
      <td>182879.773110</td>
    </tr>
    <tr>
      <th>14</th>
      <td>281463.647389</td>
      <td>183158.780998</td>
    </tr>
    <tr>
      <th>15</th>
      <td>281742.655276</td>
      <td>183437.788885</td>
    </tr>
    <tr>
      <th>16</th>
      <td>282021.663163</td>
      <td>183716.796772</td>
    </tr>
    <tr>
      <th>17</th>
      <td>282300.671050</td>
      <td>183995.804660</td>
    </tr>
    <tr>
      <th>18</th>
      <td>282579.678938</td>
      <td>184274.812547</td>
    </tr>
    <tr>
      <th>19</th>
      <td>282858.686825</td>
      <td>184553.820434</td>
    </tr>
    <tr>
      <th>20</th>
      <td>283137.694712</td>
      <td>184832.828322</td>
    </tr>
    <tr>
      <th>21</th>
      <td>283416.702600</td>
      <td>185111.836209</td>
    </tr>
    <tr>
      <th>22</th>
      <td>283695.710487</td>
      <td>185390.844096</td>
    </tr>
    <tr>
      <th>23</th>
      <td>283974.718374</td>
      <td>185669.851984</td>
    </tr>
    <tr>
      <th>24</th>
      <td>284253.726262</td>
      <td>185948.859871</td>
    </tr>
    <tr>
      <th>25</th>
      <td>284532.734149</td>
      <td>186227.867758</td>
    </tr>
  </tbody>
</table>
</div>



This dents our numbers by ~$55k for women at the pessimistic end, and ~$80k for men, but we're still solidly in the green.

## General Statistics about Education

A sort of addendum. Here I visualise some interesting data before we go into the final section.


```python
mapped_df = mincer_df.copy()
mapped_df['FOS'] = mapped_df['CIP2021'].map(CIP2021)
mapped_df['Degree'] = mapped_df['SSGRAD'].map(SSGRAD)
mapped_df['gen'] = mapped_df['Gender'].map(GENDER)
mapped_df['gen'] = mapped_df['Gender'].map(GENDER)
mapped_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PPSORT</th>
      <th>AGEGRP</th>
      <th>CIP2021</th>
      <th>EmpIn</th>
      <th>Gender</th>
      <th>LFACT</th>
      <th>PR</th>
      <th>SSGRAD</th>
      <th>log_wage</th>
      <th>age_mid</th>
      <th>edu_years</th>
      <th>exp_years</th>
      <th>FOS</th>
      <th>Degree</th>
      <th>gen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>11</td>
      <td>8</td>
      <td>12000</td>
      <td>1</td>
      <td>3</td>
      <td>35</td>
      <td>6</td>
      <td>9.392662</td>
      <td>37</td>
      <td>14</td>
      <td>17</td>
      <td>Architecture/engineering/trades</td>
      <td>College/CEGEP</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>16</td>
      <td>4</td>
      <td>61000</td>
      <td>1</td>
      <td>13</td>
      <td>35</td>
      <td>11</td>
      <td>11.018629</td>
      <td>62</td>
      <td>18</td>
      <td>38</td>
      <td>Social sciences &amp; law</td>
      <td>Masters</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>12</td>
      <td>13</td>
      <td>25000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>4</td>
      <td>10.126631</td>
      <td>42</td>
      <td>12</td>
      <td>24</td>
      <td>No postsecondary degree</td>
      <td>HS</td>
      <td>Man</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>13</td>
      <td>5</td>
      <td>130000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>8</td>
      <td>11.775290</td>
      <td>47</td>
      <td>16</td>
      <td>25</td>
      <td>Business/management/public admin</td>
      <td>Bachelor</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21</td>
      <td>10</td>
      <td>5</td>
      <td>63000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>11</td>
      <td>11.050890</td>
      <td>32</td>
      <td>18</td>
      <td>8</td>
      <td>Business/management/public admin</td>
      <td>Masters</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>378842</th>
      <td>980851</td>
      <td>11</td>
      <td>8</td>
      <td>180000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>12</td>
      <td>12.100712</td>
      <td>37</td>
      <td>21</td>
      <td>10</td>
      <td>Architecture/engineering/trades</td>
      <td>Doctorate</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>378843</th>
      <td>980852</td>
      <td>12</td>
      <td>13</td>
      <td>49000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>10.799576</td>
      <td>42</td>
      <td>10</td>
      <td>26</td>
      <td>No postsecondary degree</td>
      <td>No certificate</td>
      <td>Man</td>
    </tr>
    <tr>
      <th>378844</th>
      <td>980856</td>
      <td>12</td>
      <td>10</td>
      <td>110000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>6</td>
      <td>11.608236</td>
      <td>42</td>
      <td>14</td>
      <td>22</td>
      <td>Health</td>
      <td>College/CEGEP</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>378846</th>
      <td>980862</td>
      <td>16</td>
      <td>13</td>
      <td>33000</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>4</td>
      <td>10.404263</td>
      <td>62</td>
      <td>12</td>
      <td>44</td>
      <td>No postsecondary degree</td>
      <td>HS</td>
      <td>Woman</td>
    </tr>
    <tr>
      <th>378848</th>
      <td>980866</td>
      <td>10</td>
      <td>5</td>
      <td>130000</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>6</td>
      <td>11.775290</td>
      <td>32</td>
      <td>14</td>
      <td>12</td>
      <td>Business/management/public admin</td>
      <td>College/CEGEP</td>
      <td>Man</td>
    </tr>
  </tbody>
</table>
<p>182864 rows × 15 columns</p>
</div>




```python
sns.set_theme(style="whitegrid", font="Monospace", font_scale=1, rc={'figure.figsize': (14, 8)})
median_deg = mapped_df.groupby('Degree')['EmpIn'].median().sort_values()
g = sns.barplot(median_deg, palette="Blues_d", legend=False)
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_xlabel('Degree')
g.set_ylabel('Median Employment Income (CAD$)')
g.set_title('Median Income by Degree', y=1.02)
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_39_1.png)
    


We see a pretty linear increase as levels of education increase, but not all degree jumps are equal. Getting a bachelor (vs HS) increases your earnings by roughly 80%, where a masters merely gives you ~$20k on top of that, but for a 2 year investment. It may be justified if you're getting a doctorate, at the cost of an additional 3 years of study but with nearly double the median earnings of a bachelor - it will not take long to earn the money back.  The difference in trades and a bachelor expressed like this is minor, worse when you consider the lower time expenditure and debt burden; however, we've already calculated the NPV to find that a bachelor still makes sense.


```python
median_fos = mapped_df.groupby('FOS')['EmpIn'].median().sort_values()
g = sns.barplot(median_fos, palette='Blues_d')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_xlabel('Field of Study')
g.set_ylabel('Median Employment Income (CAD$)')
g.set_title('Median Income by Field of Study', y=1.02)
plt.show()
```
    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_41_1.png)
    


There is a wide variation in median earnings by field of study as well - STEM (and, curiously, 'education') pays the most and the arts and humanities pay the least. Education can be explained by the fact that teachers are paid handsomely in Ontario ['with salaries ranging from $65,000 to $110,000 per year'](https://www.remitly.com/blog/en-ca/jobs-and-careers/teacher-salary/). As a category, social sciences & law may make sense academically but does complicate our analysis: law degrees and sociology majors will likely get paid quite differently.


```python
gen_median_deg = mapped_df.groupby(['gen', 'Degree'])['EmpIn'].median().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

genders = ['Woman', 'Man']

for ax, gender in zip(axes, genders):
    data = gen_median_deg[gender]
    sns.barplot(x=data.index, y=data.values, ax=ax, palette='Blues_d', hue=data.index)
    ax.set_title(gender)
    ax.set_xlabel('')
    ax.set_ylabel('Median Employment Income (CAD$)' if gender == 'Woman' else '')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig.suptitle('Median Income by Degree and Gender', y=1.02)
plt.tight_layout()
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_43_1.png)
    



```python
gap_deg = gen_median_deg.unstack(level=0)
gap_deg['Gap'] = ((gap_deg['Woman'] / gap_deg['Man']) * 100)
g = sns.barplot(gap_deg['Gap'].sort_values(ascending=False), palette='Blues_d')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_xlabel('Field of Study')
g.set_ylabel('Gender Gap (Female Earnings as % of Men)')
g.set_title('Gender Gap by Degree', y=1.02)
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_44_1.png)
    


This data is unsurprising. The more educated women are, the more they surmount the gender pay gap - the situation, as common-sense would tell you, is worst in the trades. What's surprising is that women without any certification (in terms of gender gap relative to male counterparts, not absolute income) do better in this regard. Of course, 'No certificate' is an odd category, likely a small number of immigrants or refugees who don't have any Canadian certifications, and you shouldn't read too much into it. The reasons for the pay gap declining by degree are manifold, and beyond the scope of this post.


```python
gen_median_fos = mapped_df.groupby(['gen', 'FOS'])['EmpIn'].median()
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

genders = ['Woman', 'Man']

for ax, gender in zip(axes, genders):
    data = gen_median_fos[gender].sort_values()
    sns.barplot(x=data.index, y=data.values, ax=ax, palette='Blues_d', hue=data.index)
    ax.set_title(gender)
    ax.set_xlabel('')
    ax.set_ylabel('Median Employment Income (CAD$)' if gender == 'Woman' else '')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig.suptitle('Median Income by Field of Study and Gender', y=1.02)
plt.tight_layout()
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_46_1.png)
    



```python
gap = gen_median_fos.unstack(level=0)
gap['Gap'] = ((gap['Woman'] / gap['Man']) * 100)
g = sns.barplot(gap['Gap'].sort_values(ascending=False), palette='Blues_d')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_xlabel('Field of Study')
g.set_ylabel('Gender Pay Gap (Female Earnings as % of Men)')
g.set_title('Gender Gap by Field of Study', y=1.02)
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_47_1.png)
    


Women in the performing arts and 'comm', and STEM suffer the lowest gender pay gaps. The worst pay gap is in 'personal/protective/transport services', the non-degree category, and social sciences & law. The latter may be explained primarily through the law subcategory. Lawyer jobs are extremely demanding ('greedy') and women tend to disprefer such jobs; female representation in that category, therefore, will likely be in the lower-paying degrees in the social sciences, therefore skewing the category's gap. 

## Field of Study Comparisons

Back to the plot. Rationally speaking, there's really no argument against even maximal changes to the loan and grant ratio for OSAP. Of course, (future) students are *relatively* worse off and that is a negative, regardless of whether you think it's a sufficient criticism (for that, you'd need to analyse a whole lot more than just ROI, including Ontario's fiscal space). But is that really all there is to it? Let's break down the analysis into more granular terms to get a clearer picture before our final conclusion. We will calculate NPV by the following categories:

1. Field of Study
2. Gender

We could also do so by degree, but degree finances get more and more complicated as you go up the chain - doctorates being worst in this regard. Such an analysis would require a separate notebook.

I will do two separate analyses: one will compare a bachelors to a hs diploma, the other will compare a bachelors to a trades degree, by field of study.

Both analyses will use a pessimistic scenario for osap ratios: 100% loans.


```python
# Filters for post-secondary degrees
ps_df = mapped_df.copy()

# This model doesn't include edu_years to prevent multicollinearity
model_fos = sm.OLS.from_formula('log_wage ~ edu_years + C(FOS)*exp_years + C(FOS)*I(exp_years**2) + C(Gender)', data=ps_df)
results_fos = model_fos.fit()

results_fos.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>log_wage</td>     <th>  R-squared:         </th>  <td>   0.113</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.112</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   626.4</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 13 Apr 2026</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:07:43</td>     <th>  Log-Likelihood:    </th> <td>-3.3861e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>182864</td>      <th>  AIC:               </th>  <td>6.773e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>182826</td>      <th>  BIC:               </th>  <td>6.777e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    37</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                                   <td></td>                                     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                                          <td>    7.6506</td> <td>    0.105</td> <td>   72.889</td> <td> 0.000</td> <td>    7.445</td> <td>    7.856</td>
</tr>
<tr>
  <th>C(FOS)[T.Architecture/engineering/trades]</th>                          <td>    0.0265</td> <td>    0.101</td> <td>    0.261</td> <td> 0.794</td> <td>   -0.172</td> <td>    0.225</td>
</tr>
<tr>
  <th>C(FOS)[T.Business/management/public admin]</th>                         <td>    0.0067</td> <td>    0.100</td> <td>    0.067</td> <td> 0.947</td> <td>   -0.190</td> <td>    0.203</td>
</tr>
<tr>
  <th>C(FOS)[T.Education]</th>                                                <td>   -0.2288</td> <td>    0.122</td> <td>   -1.880</td> <td> 0.060</td> <td>   -0.467</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(FOS)[T.Health]</th>                                                   <td>    0.0384</td> <td>    0.102</td> <td>    0.376</td> <td> 0.707</td> <td>   -0.162</td> <td>    0.239</td>
</tr>
<tr>
  <th>C(FOS)[T.Humanities]</th>                                               <td>   -0.3519</td> <td>    0.112</td> <td>   -3.130</td> <td> 0.002</td> <td>   -0.572</td> <td>   -0.132</td>
</tr>
<tr>
  <th>C(FOS)[T.Math/CS/info]</th>                                             <td>   -0.0971</td> <td>    0.110</td> <td>   -0.885</td> <td> 0.376</td> <td>   -0.312</td> <td>    0.118</td>
</tr>
<tr>
  <th>C(FOS)[T.No postsecondary degree]</th>                                  <td>   -0.6518</td> <td>    0.099</td> <td>   -6.571</td> <td> 0.000</td> <td>   -0.846</td> <td>   -0.457</td>
</tr>
<tr>
  <th>C(FOS)[T.Personal/protective/transport services]</th>                   <td>   -0.1255</td> <td>    0.112</td> <td>   -1.118</td> <td> 0.263</td> <td>   -0.345</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(FOS)[T.Physical/life sciences & tech]</th>                            <td>   -0.5344</td> <td>    0.107</td> <td>   -4.984</td> <td> 0.000</td> <td>   -0.745</td> <td>   -0.324</td>
</tr>
<tr>
  <th>C(FOS)[T.Social sciences & law]</th>                                    <td>   -0.1646</td> <td>    0.101</td> <td>   -1.623</td> <td> 0.105</td> <td>   -0.363</td> <td>    0.034</td>
</tr>
<tr>
  <th>C(FOS)[T.Visual/performing arts & comm]</th>                            <td>   -0.5159</td> <td>    0.114</td> <td>   -4.528</td> <td> 0.000</td> <td>   -0.739</td> <td>   -0.293</td>
</tr>
<tr>
  <th>C(Gender)[T.2]</th>                                                     <td>    0.3653</td> <td>    0.008</td> <td>   46.602</td> <td> 0.000</td> <td>    0.350</td> <td>    0.381</td>
</tr>
<tr>
  <th>edu_years</th>                                                          <td>    0.1216</td> <td>    0.003</td> <td>   45.412</td> <td> 0.000</td> <td>    0.116</td> <td>    0.127</td>
</tr>
<tr>
  <th>exp_years</th>                                                          <td>    0.1069</td> <td>    0.011</td> <td>   10.079</td> <td> 0.000</td> <td>    0.086</td> <td>    0.128</td>
</tr>
<tr>
  <th>C(FOS)[T.Architecture/engineering/trades]:exp_years</th>                <td>    0.0014</td> <td>    0.011</td> <td>    0.126</td> <td> 0.900</td> <td>   -0.020</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(FOS)[T.Business/management/public admin]:exp_years</th>               <td>   -0.0074</td> <td>    0.011</td> <td>   -0.672</td> <td> 0.501</td> <td>   -0.029</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(FOS)[T.Education]:exp_years</th>                                      <td>    0.0234</td> <td>    0.013</td> <td>    1.795</td> <td> 0.073</td> <td>   -0.002</td> <td>    0.049</td>
</tr>
<tr>
  <th>C(FOS)[T.Health]:exp_years</th>                                         <td>   -0.0220</td> <td>    0.011</td> <td>   -1.955</td> <td> 0.051</td> <td>   -0.044</td> <td> 5.06e-05</td>
</tr>
<tr>
  <th>C(FOS)[T.Humanities]:exp_years</th>                                     <td>   -0.0027</td> <td>    0.012</td> <td>   -0.216</td> <td> 0.829</td> <td>   -0.027</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(FOS)[T.Math/CS/info]:exp_years</th>                                   <td>    0.0123</td> <td>    0.012</td> <td>    1.023</td> <td> 0.306</td> <td>   -0.011</td> <td>    0.036</td>
</tr>
<tr>
  <th>C(FOS)[T.No postsecondary degree]:exp_years</th>                        <td>    0.0130</td> <td>    0.011</td> <td>    1.200</td> <td> 0.230</td> <td>   -0.008</td> <td>    0.034</td>
</tr>
<tr>
  <th>C(FOS)[T.Personal/protective/transport services]:exp_years</th>         <td>   -0.0107</td> <td>    0.012</td> <td>   -0.880</td> <td> 0.379</td> <td>   -0.035</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(FOS)[T.Physical/life sciences & tech]:exp_years</th>                  <td>    0.0429</td> <td>    0.012</td> <td>    3.526</td> <td> 0.000</td> <td>    0.019</td> <td>    0.067</td>
</tr>
<tr>
  <th>C(FOS)[T.Social sciences & law]:exp_years</th>                          <td>    0.0043</td> <td>    0.011</td> <td>    0.380</td> <td> 0.704</td> <td>   -0.018</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(FOS)[T.Visual/performing arts & comm]:exp_years</th>                  <td>    0.0036</td> <td>    0.013</td> <td>    0.282</td> <td> 0.778</td> <td>   -0.021</td> <td>    0.028</td>
</tr>
<tr>
  <th>I(exp_years ** 2)</th>                                                  <td>   -0.0024</td> <td>    0.000</td> <td>  -10.098</td> <td> 0.000</td> <td>   -0.003</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(FOS)[T.Architecture/engineering/trades]:I(exp_years ** 2)</th>        <td>    0.0002</td> <td>    0.000</td> <td>    0.685</td> <td> 0.494</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Business/management/public admin]:I(exp_years ** 2)</th>       <td>    0.0004</td> <td>    0.000</td> <td>    1.651</td> <td> 0.099</td> <td>-7.67e-05</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Education]:I(exp_years ** 2)</th>                              <td>   -0.0004</td> <td>    0.000</td> <td>   -1.408</td> <td> 0.159</td> <td>   -0.001</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(FOS)[T.Health]:I(exp_years ** 2)</th>                                 <td>    0.0008</td> <td>    0.000</td> <td>    3.033</td> <td> 0.002</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Humanities]:I(exp_years ** 2)</th>                             <td>    0.0003</td> <td>    0.000</td> <td>    1.020</td> <td> 0.308</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Math/CS/info]:I(exp_years ** 2)</th>                           <td>-9.931e-05</td> <td>    0.000</td> <td>   -0.361</td> <td> 0.718</td> <td>   -0.001</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(FOS)[T.No postsecondary degree]:I(exp_years ** 2)</th>                <td>    0.0004</td> <td>    0.000</td> <td>    1.465</td> <td> 0.143</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Personal/protective/transport services]:I(exp_years ** 2)</th> <td>    0.0004</td> <td>    0.000</td> <td>    1.330</td> <td> 0.183</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Physical/life sciences & tech]:I(exp_years ** 2)</th>          <td>   -0.0008</td> <td>    0.000</td> <td>   -2.706</td> <td> 0.007</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(FOS)[T.Social sciences & law]:I(exp_years ** 2)</th>                  <td>    0.0001</td> <td>    0.000</td> <td>    0.498</td> <td> 0.619</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(FOS)[T.Visual/performing arts & comm]:I(exp_years ** 2)</th>          <td>    0.0002</td> <td>    0.000</td> <td>    0.625</td> <td> 0.532</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>142563.407</td> <th>  Durbin-Watson:     </th>  <td>   1.997</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>3512451.082</td>
</tr>
<tr>
  <th>Skew:</th>            <td>-3.625</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>        <td>23.210</td>   <th>  Cond. No.          </th>  <td>9.34e+04</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 9.34e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
hs_npvs = {}
for fos in ps_df['FOS'].dropna().unique():
    if fos == 'No postsecondary degree':
        continue
    npvs = osap_npv(ratio=0, main_model=results_fos, sec_model=results, 
                    bFos=True, fos=fos)
    hs_npvs[fos] = npvs

hs_npvs_df = pd.DataFrame.from_dict(hs_npvs, orient='index', columns=['Male', 'Female'])
hs_npvs_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
      <th>Female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Architecture/engineering/trades</th>
      <td>511710.016950</td>
      <td>357405.052259</td>
    </tr>
    <tr>
      <th>Social sciences &amp; law</th>
      <td>333757.556643</td>
      <td>233903.671114</td>
    </tr>
    <tr>
      <th>Business/management/public admin</th>
      <td>427342.998907</td>
      <td>298853.216373</td>
    </tr>
    <tr>
      <th>Math/CS/info</th>
      <td>458971.503149</td>
      <td>320803.820208</td>
    </tr>
    <tr>
      <th>Health</th>
      <td>349597.080201</td>
      <td>244896.511745</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>378718.398483</td>
      <td>265107.095080</td>
    </tr>
    <tr>
      <th>Humanities</th>
      <td>109472.619446</td>
      <td>78246.932983</td>
    </tr>
    <tr>
      <th>Personal/protective/transport services</th>
      <td>203982.747130</td>
      <td>143838.222258</td>
    </tr>
    <tr>
      <th>Physical/life sciences &amp; tech</th>
      <td>278627.998659</td>
      <td>195643.022505</td>
    </tr>
    <tr>
      <th>Visual/performing arts &amp; comm</th>
      <td>35355.603031</td>
      <td>26808.734951</td>
    </tr>
    <tr>
      <th>Agriculture/natural resources/conservation</th>
      <td>358928.782511</td>
      <td>251372.837624</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

genders = ['Male', 'Female']

for ax, gender in zip(axes, genders):
    data = hs_npvs_df[gender].sort_values()
    sns.barplot(x=data.index, y=data.values, ax=ax, palette='Blues_d', hue=data.index)
    ax.set_title(gender)
    ax.set_xlabel('')
    ax.set_ylabel('Net Present Value (CAD$)' if gender == 'Male' else '')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig.suptitle('Net Present Value of Bachelors by Field of Study and Gender', y=1.02)
plt.tight_layout()
plt.show()
```


    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_52_1.png)
    


We see an extremely vide variation in NPV as well. The counterfactual here is a high school degree. Overall, all fields of study are still a net positive, but the arts are close to having a negative ROI. 


```python
trade_npvs = {}
for fos in ps_df['FOS'].dropna().unique():
    if fos == 'No postsecondary degree':
        continue
    npvs = osap_npv(ratio=0, main_model=results_fos, sec_model=results, bFos=True, fos=fos, sec_edu=13)
    trade_npvs[fos] = npvs

trade_npvs_df = pd.DataFrame.from_dict(trade_npvs, orient='index', columns=['Male', 'Female'])
trade_npvs_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
      <th>Female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Architecture/engineering/trades</th>
      <td>429552.143647</td>
      <td>301687.835761</td>
    </tr>
    <tr>
      <th>Social sciences &amp; law</th>
      <td>251599.683340</td>
      <td>178186.454616</td>
    </tr>
    <tr>
      <th>Business/management/public admin</th>
      <td>345185.125603</td>
      <td>243135.999875</td>
    </tr>
    <tr>
      <th>Math/CS/info</th>
      <td>376813.629846</td>
      <td>265086.603709</td>
    </tr>
    <tr>
      <th>Health</th>
      <td>267439.206897</td>
      <td>189179.295247</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>296560.525180</td>
      <td>209389.878582</td>
    </tr>
    <tr>
      <th>Humanities</th>
      <td>27314.746143</td>
      <td>22529.716485</td>
    </tr>
    <tr>
      <th>Personal/protective/transport services</th>
      <td>121824.873827</td>
      <td>88121.005760</td>
    </tr>
    <tr>
      <th>Physical/life sciences &amp; tech</th>
      <td>196470.125356</td>
      <td>139925.806007</td>
    </tr>
    <tr>
      <th>Visual/performing arts &amp; comm</th>
      <td>-46802.270272</td>
      <td>-28908.481547</td>
    </tr>
    <tr>
      <th>Agriculture/natural resources/conservation</th>
      <td>276770.909208</td>
      <td>195655.621126</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

genders = ['Male', 'Female']

for ax, gender in zip(axes, genders):
    data = trade_npvs_df[gender].sort_values()
    sns.barplot(x=data.index, y=data.values, ax=ax, palette='Blues_d', hue=data.index)
    ax.set_title(gender)
    ax.set_xlabel('')
    ax.set_ylabel('Net Present Value (CAD$)' if gender == 'Male' else '')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig.suptitle('Net Present Value of Bachelors by Field of Study and Gender', y=1.02)
plt.tight_layout()
plt.show()
```

    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/osap-article/main_files/main_55_1.png)
    


We finally get our first instance of education with a negative ROI - compared to becoming a skilled tradesman, getting an arts degree is a bad investment. Humanities really toes it close to zero, but is positive with a NPV of $~20k for both genders. Beyond that however, most fields still offer a substantial (>$100k) positive ROI for both genders.

## Conclusion

In sum, education offers a positive ROI regardless of gender, grant/loan ratios, and field of study (with the sole exception of visual/performing arts vs trades) relative to a high school diploma and trades. There is wide variance between fields of study and degree type, and a substantial gender gap that also varies by FoS/Degree. 

On average, men earn ~47% more than women controlling for education and experience. This is a 'naive' gap - the entire difference is not made up solely of discrimination, since our model does not account for many variables. Generally, the gender gap decreases as women climb the education ladder and it is lowest in the Arts and STEM fields (this is just the gap, and does not account for absolute income). There is a substantial pay gap in the trades, as well. Put together, this tells us that education is likely a more important investment for women relative to men.

OSAP's recent changes make no difference to the NPV of education. Through a sensitivity analysis comparing grant/loan ratios, I found that even in the worst case scenario (100% loans), the NPV of education (compared to a counterfactual trades degree) is $270k for men and $180k for women. The OSAP ratio change actually makes very little difference at all - this is largely because 70% of federal loans are interest-free. The number does vary by field of study - visual/performing arts degrees are negative (<$10k), and humanities degrees offer an insubstantial ROI (~$20k for both genders), but most other fields offer >$100k returns. This tells us that recent or anticipated OSAP changes should not affect your college plans, except where your liquidity is questionable (i.e. OSAP does not cover 100% of college costs, and you can not make up the difference).


### Limitations

Many of these limitations are for simplicity's sake - to prevent this notebook from becoming too large. It bears listing them, however. There are also specific limitations listed in their relevant sections.

- My models factor unemployment in through its effect (drag) on income, not as a separate variable. A more complex model would handle this factor separately and properly. This could be done by modelling employment probability separately and combining *that* with income predictions.
- Experience calculation is also affected similarly - few people would have 45 years of experience at the end of their working lifetime, because that assumes continuous employment (i.e. no unemployment). 
- My models do not control for that many variables - hence the low R^2 of ~0.1. These include specific factors relevant to income, like part-time/full-time work or hours worked per week, and general factors like race or immigration status. 
- My models do not account for the weights given by PUMF. That means my model is not ideal for population-level results - this is fine for measuring relationships, but knocks my precision.
- Immigrants might deserve a separate notebook, esp. for those with foreign education.
- I use an interest rate of 6% throughout my calculations. An extension of this notebook would be to perform a sensitivity analysis.
- There are some simplifications when calculating OSAP loan annuities. There is a 6-month grace period, and the actual time horizon is 9.5 years (where I use 10). Since my cash flow models annual income, it is difficult to accomodate this - hence the simplification. There are also tax considerations vis-a-vis both income and student loan tax credits that I have ignored.

