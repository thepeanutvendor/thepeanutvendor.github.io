---
title: "Testing the Phillips Curve"
date: 2026-04-03T21:31:07+05:00
draft: false
description: "In this notebook, I try to replicate and analyse the Phillips Curve."
series: "Replicating Economic Models"
series_order: 1
---
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set_theme()
```

# Testing the Phillips Curve

The Phillips Curve argues that there is a negative correlation between inflation and unemployment - when inflation increases, unemployment decreases, and vice versa. The mechanism this occurs by is hypothesised as such:

1. Suppose that unemployment has decreased
2. Employees have more bargaining power, because of the decrease in supply of labor, and can demand higher wages
3. Higher wages lead to higher costs of production
4. These costs are passed onto consumers, leading to an increase in the general price level (inflation)

This traditional formulation was empirically disproven by the oil shock after 1973. This brought on a decade of stagflation, where increases in inflation did not correlate with decreases in unemployment. It was replaced with the Expectations Augmented Phillips Curve, which provides a fuller picture, taking the long-run and worker expectations into account:

1. In the long run, employees will see their real wages decline
2. This will incite them to demand greater wages
3. This will force firms to lay off workers
4. Employment will return to its natural rate, but inflation will still be high


## The Traditional Phillips Curve

By plotting unemployment and inflation data, we can get a general idea of their correlation. I will use FRED's annual inflation data for the US, not seasonally adjusted, and its monthly unemployment data, seasonally adjusted.


```python
inf = pd.read_csv('inf.csv', index_col='observation_date', parse_dates=True).rename(columns={'FPCPITOTLZGUSA': 'inflation_rate'})
inf
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
      <th>inflation_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>1.457976</td>
    </tr>
    <tr>
      <th>1961-01-01</th>
      <td>1.070724</td>
    </tr>
    <tr>
      <th>1962-01-01</th>
      <td>1.198773</td>
    </tr>
    <tr>
      <th>1963-01-01</th>
      <td>1.239669</td>
    </tr>
    <tr>
      <th>1964-01-01</th>
      <td>1.278912</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>1.233584</td>
    </tr>
    <tr>
      <th>2021-01-01</th>
      <td>4.697859</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>8.002800</td>
    </tr>
    <tr>
      <th>2023-01-01</th>
      <td>4.116338</td>
    </tr>
    <tr>
      <th>2024-01-01</th>
      <td>2.949525</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 1 columns</p>
</div>




```python
unemp = pd.read_csv('unemp.csv', index_col='observation_date', parse_dates=True).rename(columns={'UNRATE': 'unemployment_rate'})
unemp
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
      <th>unemployment_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1948-01-01</th>
      <td>3.4</td>
    </tr>
    <tr>
      <th>1948-02-01</th>
      <td>3.8</td>
    </tr>
    <tr>
      <th>1948-03-01</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1948-04-01</th>
      <td>3.9</td>
    </tr>
    <tr>
      <th>1948-05-01</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-10-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2025-11-01</th>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2025-12-01</th>
      <td>4.4</td>
    </tr>
    <tr>
      <th>2026-01-01</th>
      <td>4.3</td>
    </tr>
    <tr>
      <th>2026-02-01</th>
      <td>4.4</td>
    </tr>
  </tbody>
</table>
<p>938 rows × 1 columns</p>
</div>



Now we have to merge the dataframes. Our unemployment data is monthly, so we have to resample it to an annual basis. Annual unemployment is calculated as the average of the 12 monthly figures.


```python
unemp_inf = pd.merge(unemp.resample('YS').mean(), inf, how='left', on='observation_date')
unemp_inf.tail(20)
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
      <th>unemployment_rate</th>
      <th>inflation_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007-01-01</th>
      <td>4.616667</td>
      <td>2.852672</td>
    </tr>
    <tr>
      <th>2008-01-01</th>
      <td>5.800000</td>
      <td>3.839100</td>
    </tr>
    <tr>
      <th>2009-01-01</th>
      <td>9.283333</td>
      <td>-0.355546</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>9.608333</td>
      <td>1.640043</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>8.933333</td>
      <td>3.156842</td>
    </tr>
    <tr>
      <th>2012-01-01</th>
      <td>8.075000</td>
      <td>2.069337</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>7.358333</td>
      <td>1.464833</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>6.158333</td>
      <td>1.622223</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>5.275000</td>
      <td>0.118627</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>4.875000</td>
      <td>1.261583</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>4.358333</td>
      <td>2.130110</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>3.891667</td>
      <td>2.442583</td>
    </tr>
    <tr>
      <th>2019-01-01</th>
      <td>3.675000</td>
      <td>1.812210</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>8.100000</td>
      <td>1.233584</td>
    </tr>
    <tr>
      <th>2021-01-01</th>
      <td>5.350000</td>
      <td>4.697859</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>3.650000</td>
      <td>8.002800</td>
    </tr>
    <tr>
      <th>2023-01-01</th>
      <td>3.625000</td>
      <td>4.116338</td>
    </tr>
    <tr>
      <th>2024-01-01</th>
      <td>4.025000</td>
      <td>2.949525</td>
    </tr>
    <tr>
      <th>2025-01-01</th>
      <td>4.263636</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2026-01-01</th>
      <td>4.350000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Our inflation data is missing for many dates. We must drop the rows where we don't have matching unemployment and inflation data.


```python
unemp_inf = unemp_inf.dropna()
unemp_inf
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
      <th>unemployment_rate</th>
      <th>inflation_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>5.541667</td>
      <td>1.457976</td>
    </tr>
    <tr>
      <th>1961-01-01</th>
      <td>6.691667</td>
      <td>1.070724</td>
    </tr>
    <tr>
      <th>1962-01-01</th>
      <td>5.566667</td>
      <td>1.198773</td>
    </tr>
    <tr>
      <th>1963-01-01</th>
      <td>5.641667</td>
      <td>1.239669</td>
    </tr>
    <tr>
      <th>1964-01-01</th>
      <td>5.158333</td>
      <td>1.278912</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>8.100000</td>
      <td>1.233584</td>
    </tr>
    <tr>
      <th>2021-01-01</th>
      <td>5.350000</td>
      <td>4.697859</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>3.650000</td>
      <td>8.002800</td>
    </tr>
    <tr>
      <th>2023-01-01</th>
      <td>3.625000</td>
      <td>4.116338</td>
    </tr>
    <tr>
      <th>2024-01-01</th>
      <td>4.025000</td>
      <td>2.949525</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 2 columns</p>
</div>




```python
sns.lineplot(unemp_inf)
```




    <Axes: xlabel='observation_date'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_9_1.png "Fig 1: Unemployment and Inflation")
    


What's cool about this specific data is that we can trace the history of economic thought, specifically the Phillips' Curve, through it. If we look at the 1960-70 period, we see the negative correlation the traditional model argues for. 1970 onwards, we see that correlation completely collapse.


```python
sixties = unemp_inf['1960-01-01':'1967-01-01']
sns.lineplot(sixties)
```




    <Axes: xlabel='observation_date'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_11_1.png "Fig 2: Unemployment and Inflation in the 60s" )
    



```python
sixties['unemployment_rate'].corr(sixties['inflation_rate']) 
```




    np.float64(-0.8812049507014807)



We see a reasonably strong negative correlation, as Bill Phillips demonstrated.


```python
sns.regplot(x=sixties['inflation_rate'], y=sixties['unemployment_rate'])
```




    <Axes: xlabel='inflation_rate', ylabel='unemployment_rate'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_14_1.png "Fig 3: A Regression Plot of the Same")
    


## Stagflation


```python
seventies = unemp_inf['1970-01-01':'1980-01-01']
sns.lineplot(seventies)
```




    <Axes: xlabel='observation_date'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_16_1.png "Fig 4: Unemployment and Inflation in the 70s")
    



```python
seventies['unemployment_rate'].corr(seventies['inflation_rate'])
```




    np.float64(0.2663010944577987)



We see a weak positive correlation. Simultaneous highs of inflation and unemployment seemed to disprove the Traditional Phillips Curve.


```python
sns.regplot(x=seventies['inflation_rate'], y=seventies['unemployment_rate'])
```




    <Axes: xlabel='inflation_rate', ylabel='unemployment_rate'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_19_1.png "Fig 5: A Regression Plot of the Same")
    


## The Short-run

Economists do not hold the traditional Phillips Curve to be false, but merely a short-run theory. Now I'm going to test if this is true: where short-run is equal to one year, does a strong negative correlation between unemployment and inflation hold?

To do this, we need to get monthly YoY inflation data. FRED gives us 'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average'. We will need to convert this to percentage changes, but it should work.


```python
cpi_monthly = pd.read_csv('inf_monthly.csv', index_col='observation_date', parse_dates=True).rename(columns={'CPIAUCSL': 'CPI'})
inf_monthly = (cpi_monthly.pct_change(12) * 100).dropna().rename(columns={'CPI': 'inflation_rate'})
inf_monthly
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
      <th>inflation_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1948-01-01</th>
      <td>10.242086</td>
    </tr>
    <tr>
      <th>1948-02-01</th>
      <td>9.481961</td>
    </tr>
    <tr>
      <th>1948-03-01</th>
      <td>6.818182</td>
    </tr>
    <tr>
      <th>1948-04-01</th>
      <td>8.272727</td>
    </tr>
    <tr>
      <th>1948-05-01</th>
      <td>9.384966</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-09-01</th>
      <td>3.022572</td>
    </tr>
    <tr>
      <th>2025-11-01</th>
      <td>2.696444</td>
    </tr>
    <tr>
      <th>2025-12-01</th>
      <td>2.653304</td>
    </tr>
    <tr>
      <th>2026-01-01</th>
      <td>2.391201</td>
    </tr>
    <tr>
      <th>2026-02-01</th>
      <td>2.434004</td>
    </tr>
  </tbody>
</table>
<p>937 rows × 1 columns</p>
</div>



Now that we have our data, we can get the average negative correlation between inflation and unemployment over twelve-month periods.


```python
unemp_inf_monthly = pd.merge(unemp, inf_monthly, 'right', on='observation_date')
unemp_inf_monthly.dropna()
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
      <th>unemployment_rate</th>
      <th>inflation_rate</th>
    </tr>
    <tr>
      <th>observation_date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1948-01-01</th>
      <td>3.4</td>
      <td>10.242086</td>
    </tr>
    <tr>
      <th>1948-02-01</th>
      <td>3.8</td>
      <td>9.481961</td>
    </tr>
    <tr>
      <th>1948-03-01</th>
      <td>4.0</td>
      <td>6.818182</td>
    </tr>
    <tr>
      <th>1948-04-01</th>
      <td>3.9</td>
      <td>8.272727</td>
    </tr>
    <tr>
      <th>1948-05-01</th>
      <td>3.5</td>
      <td>9.384966</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-09-01</th>
      <td>4.4</td>
      <td>3.022572</td>
    </tr>
    <tr>
      <th>2025-11-01</th>
      <td>4.5</td>
      <td>2.696444</td>
    </tr>
    <tr>
      <th>2025-12-01</th>
      <td>4.4</td>
      <td>2.653304</td>
    </tr>
    <tr>
      <th>2026-01-01</th>
      <td>4.3</td>
      <td>2.391201</td>
    </tr>
    <tr>
      <th>2026-02-01</th>
      <td>4.4</td>
      <td>2.434004</td>
    </tr>
  </tbody>
</table>
<p>937 rows × 2 columns</p>
</div>




```python
uninf_corr = unemp_inf_monthly.groupby(unemp_inf_monthly.index.year).corr().iloc[0::2,-1].unstack()
sns.lineplot(uninf_corr)
```




    <Axes: xlabel='observation_date'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_24_1.png "Fig 6: Correlation b/w Unemployment and Inflation Within Each Year")
    



```python
uninf_corr.mean()
```




    unemployment_rate   -0.139759
    dtype: float64



This shows us that, within 12-month periods, there is no strong correlation between unemployment and inflation. If unemployment rises in a month, inflation does not rise *simultaneously*. 

The biggest flaw with this analysis is, of course, that it fails to account for a time gap. Moreover, within 12-month periods there are only twelve total samples; averaging these does not make the reults especially meaningful. 

Instead of intra-short-run periods, let's try a 3-month gap/lag between unemployment and inflation - how does unemployment rising in one month correlate with inflation three months on?

## Adding a Lag


```python
lagged_unempinf_monthly = unemp_inf_monthly.copy()
lagged_unempinf_monthly['inflation_rate'] = unemp_inf_monthly['inflation_rate'].shift(-3).dropna()
lagged_unempinf_monthly['unemployment_rate'].corr(lagged_unempinf_monthly['inflation_rate'])
```




    np.float64(0.04095703974523833)



A weak positive correlation. Let's try generalising this function.


```python
lags = range(24) # two year range
lag_correlations = {}

lagged_unempinf_monthly = unemp_inf_monthly.copy()

for lag in lags:
    lagged_unempinf_monthly['inflation_rate'] = unemp_inf_monthly['inflation_rate'].shift(-lag)
    lag_correlations[lag] = lagged_unempinf_monthly['unemployment_rate'].corr(lagged_unempinf_monthly['inflation_rate'])

pd.DataFrame.from_dict(lag_correlations, orient='index', columns=['correlation'])
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
      <th>correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.053951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.048228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.044783</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.040957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.040465</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.042552</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.046893</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.054609</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.063986</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.075217</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.088234</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.102423</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.115654</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.124849</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.133213</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.141117</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.148507</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.155698</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.162488</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.166092</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.167879</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.168585</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.167955</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.166077</td>
    </tr>
  </tbody>
</table>
</div>



The positive correlation seems to grow by lag - meaning an increase in unemployment is correlated with an increase in inflation, generally, 1-24 months down the road. This seems counterintuitive.

One possible reason for this is that we average over the entire 1948-2026 period. Let's try breaking it down into different economic periods.


```python
eras = {
    'pre_stagflation': ('1960-01-01', '1969-12-01'),
    'stagflation':     ('1970-01-01', '1983-12-01'),
    'great_moderation':('1984-01-01', '2007-12-01'),
    'post_gfc':        ('2008-01-01', '2026-02-01'),
}

lags = range(25) # two year range
lag_correlations = []
index = 0

for era, periods in eras.items():
    era_data = unemp_inf_monthly[periods[0]:periods[1]].dropna()
    for lag in lags:
        lag_correlations.append([era, lag, era_data['unemployment_rate'].corr(era_data['inflation_rate'].shift(-lag))])
        index += 1

lag_correlations_df = pd.DataFrame(lag_correlations, columns=['Era', 'Lag', 'Correlation'])
lag_correlations_df
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
      <th>Era</th>
      <th>Lag</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pre_stagflation</td>
      <td>0</td>
      <td>-0.807014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pre_stagflation</td>
      <td>1</td>
      <td>-0.816310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pre_stagflation</td>
      <td>2</td>
      <td>-0.827009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pre_stagflation</td>
      <td>3</td>
      <td>-0.837270</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pre_stagflation</td>
      <td>4</td>
      <td>-0.847746</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>post_gfc</td>
      <td>20</td>
      <td>0.076844</td>
    </tr>
    <tr>
      <th>96</th>
      <td>post_gfc</td>
      <td>21</td>
      <td>0.071699</td>
    </tr>
    <tr>
      <th>97</th>
      <td>post_gfc</td>
      <td>22</td>
      <td>0.063336</td>
    </tr>
    <tr>
      <th>98</th>
      <td>post_gfc</td>
      <td>23</td>
      <td>0.050598</td>
    </tr>
    <tr>
      <th>99</th>
      <td>post_gfc</td>
      <td>24</td>
      <td>0.027397</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>



This gets us the correlation between inflation and unemployment, by era, across time lags from 0 to 23 months apart. Let's view this data era-by-era.


```python
sns.lineplot(x=lag_correlations_df['Lag'], y=lag_correlations_df['Correlation'], hue=lag_correlations_df['Era'])
```




    <Axes: xlabel='Lag', ylabel='Correlation'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_34_1.png "Fig 7: Correlation b/w Unemployment and Inflation, by Lag and Era")
    



```python
lag_correlations_df.groupby('Era').apply(lambda x : x.nsmallest(1, 'Correlation')) # The lowest correlation and its = lag by era
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
      <th></th>
      <th>Lag</th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>Era</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>great_moderation</th>
      <th>62</th>
      <td>12</td>
      <td>0.061450</td>
    </tr>
    <tr>
      <th>post_gfc</th>
      <th>75</th>
      <td>0</td>
      <td>-0.420974</td>
    </tr>
    <tr>
      <th>pre_stagflation</th>
      <th>8</th>
      <td>8</td>
      <td>-0.859855</td>
    </tr>
    <tr>
      <th>stagflation</th>
      <th>37</th>
      <td>12</td>
      <td>-0.408968</td>
    </tr>
  </tbody>
</table>
</div>



The lessons of this analysis are interesting. For one, we see that in all eras save for the great moderation, we have a lag that corresponds to a reasonable negative correlation between inflation and unemployment. Save for the post great recession era, the lag between which changes in unemployment are most negatively correlated with changes in inflation seems to be between 8 to 12 months. 

The pre-stagflation era tells a simple story: unemployment is strongly negatively correlated with inflation at any lag, but especially around 8 months in.

The stagflation era does not turn out to be the empirical disprover it seemed to be: at a lag of 8 months, inflation and unemployment correlate with an n=-0.408968. In the social sciences, this is a moderately strong correlation.

The great moderation era is the only one where inflation and unemployment have no link at any lag. This era is associated with prudent monetary policy, most notably, with milder business cycles and generally less volatility in economic variables like inflation and unemployment.

The post-gfc eras 'strongest' lag is 0 months. That is to say, changes in unemployment *immediately* affect inflation. This was also a complicated period: the main factors in this anomaly are likely the Great Recession and its aftermath, and COVID.

## What explains the GFC?


```python
sns.lineplot(unemp_inf[eras['post_gfc'][0]:eras['post_gfc'][1]])
```




    <Axes: xlabel='observation_date'>




    
![png](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/replicating-phillips-curve/main_files/main_38_1.png "Fig 8: Unemployment and Inflation during the Great Recession and Aftermath")
    


Inflation and unemployment both started off high in the GFC era. As the economy plunged into recession, the economy suffered a collapse in employment and deflation - simultaneously. This is likely due to the special nature of the crisis that occurred in 2008. 

In the aftermath and recovery, unemployment came down from its heights while inflation stayed stable. This is because the economy was operating below full employment, or at an unemployment gap. Closing this gap between current unemployment and the natural rate of unemployment should not generate inflationary pressure, and did not up till 2015, where we see a simultaneous increase in inflation and a continued decrease in unemployment past pre-crisis levels.

This decrease continued up till COVID. COVID was an especially unique crisis. The economy suffered highs in inflation you wouldn't usually see during a period of recession, because of supply-side issues caused by the pandemic. There is a brief lag here, which probably reflects the time between lockdowns -> supply-side issues. The collapse in inflation and unemployment, and the lag between those too, can be put down to the end of lockdowns, (the resolution of) supply-side issues, and also fiscal stimulus.

## Applying OLS Regression

A linear regression involves fitting a curve to a bunch of data points to see the relationship betweeen two variables, or in our case unemployment and inflation. Invariably, there will be some level of error (or residual) in each prediction: the difference between our predicted value of y given x and the actual value of y in a given data-point of x. An OLS regression aims to minimise the sum of each such residual (squared, to avoid weighting negative and positive residuals differently) between our data points and the curve we're fitting onto them.

The main numbers we care for are:
1. R-squared, or how much of the variation in inflation is explained by unemployment
2. coef, or slope, of unemployment_rate
3. P>|t| of unemployment_rate, which measures whether the relationship is statistically significant (p<0.05)


```python
x = unemp_inf_monthly.unemployment_rate
x = sm.add_constant(x)
y = unemp_inf_monthly.inflation_rate

result = sm.OLS(y, x).fit()

print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         inflation_rate   R-squared:                       0.003
    Model:                            OLS   Adj. R-squared:                  0.002
    Method:                 Least Squares   F-statistic:                     2.729
    Date:                Fri, 03 Apr 2026   Prob (F-statistic):             0.0988
    Time:                        20:50:14   Log-Likelihood:                -2315.4
    No. Observations:                 937   AIC:                             4635.
    Df Residuals:                     935   BIC:                             4644.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const                 3.0046      0.325      9.245      0.000       2.367       3.642
    unemployment_rate     0.0908      0.055      1.652      0.099      -0.017       0.199
    ==============================================================================
    Omnibus:                      210.433   Durbin-Watson:                   0.025
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              414.374
    Skew:                           1.295   Prob(JB):                     1.05e-90
    Kurtosis:                       4.977   Cond. No.                         21.1
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


An r^2 of 0.003 tells us the same story as our earliest analysis: there is no correlation between those two variables. The slope is positive and P>|t| tells us it's statistically insignificant. But what about by-era analysis?

### Pre-Stagflation


```python
x = unemp_inf_monthly['1960-01-01':'1969-12-01'].unemployment_rate
x = sm.add_constant(x)
y = unemp_inf_monthly['1960-01-01':'1969-12-01'].inflation_rate

result = sm.OLS(y, x).fit()

print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         inflation_rate   R-squared:                       0.651
    Model:                            OLS   Adj. R-squared:                  0.648
    Method:                 Least Squares   F-statistic:                     220.4
    Date:                Fri, 03 Apr 2026   Prob (F-statistic):           9.18e-29
    Time:                        20:50:14   Log-Likelihood:                -152.39
    No. Observations:                 120   AIC:                             308.8
    Df Residuals:                     118   BIC:                             314.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const                 7.6044      0.364     20.905      0.000       6.884       8.325
    unemployment_rate    -1.1027      0.074    -14.845      0.000      -1.250      -0.956
    ==============================================================================
    Omnibus:                       10.070   Durbin-Watson:                   0.120
    Prob(Omnibus):                  0.007   Jarque-Bera (JB):               10.964
    Skew:                           0.738   Prob(JB):                      0.00416
    Kurtosis:                       2.888   Cond. No.                         23.4
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


Here we see a strong correlation, as expected, an r^2 of 0.651. A negative slop of magnitute 1.1027, indicating a 1.1% fall in inflation with a 1% rise in unemployment. The relationship is highly statistically significant.

### Stagflation


```python
x = unemp_inf_monthly['1970-01-01':'1983-12-01'].unemployment_rate
x = sm.add_constant(x)
y = unemp_inf_monthly['1970-01-01':'1983-12-01'].inflation_rate

result = sm.OLS(y, x).fit()

print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         inflation_rate   R-squared:                       0.007
    Model:                            OLS   Adj. R-squared:                  0.001
    Method:                 Least Squares   F-statistic:                     1.124
    Date:                Fri, 03 Apr 2026   Prob (F-statistic):              0.291
    Time:                        20:50:14   Log-Likelihood:                -432.89
    No. Observations:                 168   AIC:                             869.8
    Df Residuals:                     166   BIC:                             876.0
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const                 8.5759      1.098      7.811      0.000       6.408      10.744
    unemployment_rate    -0.1649      0.156     -1.060      0.291      -0.472       0.142
    ==============================================================================
    Omnibus:                       19.229   Durbin-Watson:                   0.019
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):                9.954
    Skew:                           0.415   Prob(JB):                      0.00689
    Kurtosis:                       2.145   Cond. No.                         32.0
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


R^2 of 0.007, a slop of -0.1649, and very high P-value mean this relationship is weak and statistically insignificant.

### Great Moderation


```python
x = unemp_inf_monthly['1984-01-01':'2007-12-01'].unemployment_rate
x = sm.add_constant(x)
y = unemp_inf_monthly['1984-01-01':'2007-12-01'].inflation_rate

result = sm.OLS(y, x).fit()

print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         inflation_rate   R-squared:                       0.025
    Model:                            OLS   Adj. R-squared:                  0.022
    Method:                 Least Squares   F-statistic:                     7.431
    Date:                Fri, 03 Apr 2026   Prob (F-statistic):            0.00681
    Time:                        20:50:14   Log-Likelihood:                -427.18
    No. Observations:                 288   AIC:                             858.4
    Df Residuals:                     286   BIC:                             865.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const                 2.1690      0.350      6.189      0.000       1.479       2.859
    unemployment_rate     0.1651      0.061      2.726      0.007       0.046       0.284
    ==============================================================================
    Omnibus:                        9.285   Durbin-Watson:                   0.098
    Prob(Omnibus):                  0.010   Jarque-Bera (JB):                9.699
    Skew:                           0.448   Prob(JB):                      0.00783
    Kurtosis:                       2.916   Cond. No.                         33.1
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


R^2 of 0.025, and a positive slope and statistically significant relationship.

### Post-GFC


```python
x = unemp_inf_monthly['2008-01-01':'2026-02-01'].unemployment_rate
x = sm.add_constant(x)
y = unemp_inf_monthly['2008-01-01':'2026-02-01'].inflation_rate

result = sm.OLS(y, x).fit()

print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         inflation_rate   R-squared:                       0.177
    Model:                            OLS   Adj. R-squared:                  0.173
    Method:                 Least Squares   F-statistic:                     46.31
    Date:                Fri, 03 Apr 2026   Prob (F-statistic):           9.89e-11
    Time:                        20:50:14   Log-Likelihood:                -435.97
    No. Observations:                 217   AIC:                             875.9
    Df Residuals:                     215   BIC:                             882.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const                 4.7076      0.348     13.545      0.000       4.023       5.393
    unemployment_rate    -0.3749      0.055     -6.805      0.000      -0.484      -0.266
    ==============================================================================
    Omnibus:                       32.834   Durbin-Watson:                   0.068
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               42.705
    Skew:                           1.009   Prob(JB):                     5.33e-10
    Kurtosis:                       3.809   Cond. No.                         18.2
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


A weak R^2 of 0.177, negative slope and high statistical significance. In this era, a 1% increase in unemployment resulted in a 0.37% decrease in inflation.

Our data uses a monthly time-series. Inflation and unemployment data does not change much month from month ('autocorrelation'). OLS assumes independence, or that each data point of inflation and unemployment is unrelated to the previous data-point; this is not true of our data. We can see this in the very low Durbin-Watson values. Autocorrelation in OLS regressions tends to lead to inflated R^2 and P|t| values, both overrating the link between our variables and statistical significance. In sum, these results are not precise.

## Conclusion

In this notebook, we started by looking at US economic data generally to measure the relationship between unemployment and inflation. We found that, in the aggregate, no such relationship seemed to exist. We then broke down our data into four eras and looked at them separately, enabling a more tailored analysis. We started by simply measuring correlation by era, then added a lag too. Finally, we tried out an OLS regression (with limitations) by era. Our dataset was limited to the US exclusively, and did not vary; we did vary technique, each with its own limitations (see their sections). These limitations should be taken into account, but we can derive some conclusions with reasonable confidence.

1. For the pre-stagflation era, there was a very tight negative correlation between inflation and unemployment. This peaked at 8 months of lag - a positive change in unemployment one month correlates heavily to a negative change in inflation eight months on. Our OLS regression was not lagged, but it also gave us a slope of -1.1: a 1.1% fall in inflation leads to a 1% rise in unemployment.
2. For the stagflation era, the relationship broke down. Our OLS regression found a low R^2 and low statistical significance. However, adding a 12 month lag gave us a moderately high correlation of -0.408968, indicating the relationship existed but was quite lagged.
3. For the great moderation era, we found a statistically significant *positive* correlation between inflation and unemployment. This relationship was interesting, and is explainable quite simply: the Federal Reserve was looking at the same data as us. Under Volcker, the Fed worked rigorously to keep inflation under control - as a result, any increase would result in a tightening of policy, and increases in unemployment, and vice versa, perpetuating the positive correlation.
4. The post-GFC era got its own section in this notebook. The gist of it is that the noise in the data is because of two major events:
   1. The GFC itself, or its recovery to be exact. In the slow recovery that followed, the economy took its time returning to its NRU. In the meanwhile, inflation stayed stable and unemployment steadily decreased, which obviously weakens the relationship between our variables.
   2. COVID was a unique event; it resulted in both a demand and supply shock. With a brief period of lag, unemployment rose and so too did inflation, also complicating our data.

But the general idea is that the Phillips Curve should not simply be dismissed. Testing macroeconomic hypotheses is always complex: there actually isn't that much data, given how many factors change, country-to-country, year-to-year. Overall, though, the tight relationship in the pre-stagflation era and the fact we can explain away all the eras save for the stagflation should give us some confidence. On the other hand, in recent times the Phillips Curve has been on a losing streak - maybe its time is simply over.
