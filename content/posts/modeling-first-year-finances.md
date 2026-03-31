---
date: '2026-03-29T20:17:30+05:00'
draft: false
title: 'Modeling my first-year finances'
toc: true
category: 'Data Science'
---

[A notebook](https://github.com/thepeanutvendor/economic-analysis/blob/main/modeling-first-year-finances/main.ipynb) that models and analyses my personal finances for my first-year as an incoming University of Toronto, Faculty of Arts and Sciences student. Modeling on Pandas rather than a spreadsheet offers a few benefits on top of providing practice: it enables the application of some advanced statistical techniques not easily accessible in Excel, some advanced visualisation libraries, and so on. I have not utilised most of these features *yet*. I intend to extend this analysis further with a Monte Carlo simulation (as opposed to scenario based modeling) in the future, for instance.

## Structure

The notebook consists of cost, revenue, and net dataframes for base, pessimistic, and optimistic forecasts. In total, there are 9 such, 'core' dataframes. Costs are categorised largely according to the [UofT Financial Planner](https://planningcalc.utoronto.ca/financialPlanner/#/). So too are revenues. I estimated OSAP and UTAPS grants and loans, and zeroed scholarships and family support for simplicities sake. My main revenues aside from these are: summer work, expressed as a starting balance added to my running balance in the net dataframes; part-time work, expressed as a constant revenue source. More on these later.

<caption>Table 1.1: Net Dataframe for Pessimistic Scenario</caption>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net</th>
      <th>Running Balance</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2026-09-01</th>
      <td>1144.0</td>
      <td>6424.0</td>
    </tr>
    <tr>
      <th>2026-10-01</th>
      <td>1922.0</td>
      <td>8346.0</td>
    </tr>
    <tr>
      <th>2026-11-01</th>
      <td>-1078.0</td>
      <td>7268.0</td>
    </tr>
    <tr>
      <th>2026-12-01</th>
      <td>-1078.0</td>
      <td>6190.0</td>
    </tr>
    <tr>
      <th>2027-01-01</th>
      <td>-1000.0</td>
      <td>5190.0</td>
    </tr>
    <tr>
      <th>2027-02-01</th>
      <td>-1078.0</td>
      <td>4112.0</td>
    </tr>
    <tr>
      <th>2027-03-01</th>
      <td>-1078.0</td>
      <td>3034.0</td>
    </tr>
    <tr>
      <th>2027-04-01</th>
      <td>-1078.0</td>
      <td>1956.0</td>
    </tr>
  </tbody>
</table>
<caption>Fig 1.1: Net Monthly Revenue by Scenario</caption>

![Fig 1.1](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/modeling-first-year-finances/main_files/main_21_0.png)

After inputting all the relevant data and consolidating it into the three main models, I got to the core of my analysis: the use of aggregate metrics to analyse my forecast finances. I judged my finances by the following metrics:

 - Final balance
 - Minimum balance
 - Maximum deficit
 - Worst month
 - Months negative
 - Required buffer

<caption>Table 1.2: Metrics by Scenario</caption>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Final Balance</th>
      <th>Minimum Balance</th>
      <th>Max Deficit</th>
      <th>Worst Month</th>
      <th>Months Negative</th>
      <th>Required Buffer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Base</th>
      <td>13948.0</td>
      <td>11927.0</td>
      <td>-151.0</td>
      <td>2026-11-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Optimistic</th>
      <td>22140.0</td>
      <td>15415.0</td>
      <td>521.0</td>
      <td>2026-11-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Pessimistic</th>
      <td>1956.0</td>
      <td>1956.0</td>
      <td>-1078.0</td>
      <td>2026-11-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

I also performed a sensitivity analysis on the two main variable revenues in my budget, part-time and summer work. The intention was to determine how much I could bend these variables and stay in the green. I visualised this in heatmaps, and a feasible frontier curve.

<caption>Fig 1.2: Heatmap of Minimum Balance by Summer and Part-time Earnings</caption>

![Fig 1.2](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/modeling-first-year-finances/main_files/main_26_0.png)

<caption>Fig 1.3: Heatmap of Months Negative by Summer and Part-time Earnings</caption>

![Fig 1.3](https://raw.githubusercontent.com/thepeanutvendor/economic-analysis/refs/heads/main/modeling-first-year-finances/main_files/main_27_0.png)

<caption>Table 1.3: Feasible Frontier</caption>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Part-time Earnings</th>
      <th>Summer Earnings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>5500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>300</td>
      <td>4000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>400</td>
      <td>3000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>500</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>600</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>700</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

The final aspect of this notebook was a residence cost, and then weighted, comparison. For the former, I manually input the costs for each residence available to me to see how they would mesh with my broader budget.

<caption>Table 1.4: Key Metrics by Residence</caption>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Residence</th>
      <th>Minimum Balance</th>
      <th>Final Balance</th>
      <th>Months Negative</th>
      <th>Max Deficit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>New College</td>
      <td>2406</td>
      <td>2406.0</td>
      <td>0</td>
      <td>-3014.5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Woodsworth</td>
      <td>1956</td>
      <td>1956.0</td>
      <td>0</td>
      <td>-1078.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Knox</td>
      <td>1475</td>
      <td>1475.0</td>
      <td>0</td>
      <td>-2183.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>University College</td>
      <td>823</td>
      <td>823.0</td>
      <td>0</td>
      <td>-3806.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chestnut</td>
      <td>-2559</td>
      <td>-2560.0</td>
      <td>4</td>
      <td>-3797.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trinity</td>
      <td>-5196</td>
      <td>-5197.0</td>
      <td>4</td>
      <td>-6816.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oak</td>
      <td>-6962</td>
      <td>-6963.0</td>
      <td>4</td>
      <td>-7699.0</td>
    </tr>
  </tbody>
</table>

After that, I had three LLMs (Claude 4.6, GPT-5.3, and Deepseek) rank the (first four) viable residences across a few similar categories. I got the following results:

<caption>Table 1.5: LLM Rankings of Residences by Category</caption>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cost</th>
      <th>Room &amp; Amenities</th>
      <th>Community</th>
      <th>Food</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Woodsworth</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Knox</th>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>UC</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>New College</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

I weighted these categories according to my preferences, and then derived a ranking of residences.

<caption>Table 1.6: Weighted Ranking of Residences</caption>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weighted Ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Woodsworth</th>
      <td>1.55</td>
    </tr>
    <tr>
      <th>New College</th>
      <td>1.95</td>
    </tr>
    <tr>
      <th>UC</th>
      <td>2.75</td>
    </tr>
    <tr>
      <th>Knox</th>
      <td>3.75</td>
    </tr>
  </tbody>
</table>


## Analysis

What lessons did I derive from this? I got a viable ranking of residences, which I have used in my StarRez application. OSAP and UTAPS applications ask the applicant for estimates regarding, among other things, their starting assets and work earnings expectations throughout the school year. I have a model that can inform the values I give when asked for such details. Not to mention that I have an idea of how hard I'll need to actually work, this summer and the school year, in order to stay afloat, which is extremely useful. I also have a nice budget template, something I can refine based on my actual data even better in my second year.

Generally, my finances are viable so long as I earn either $1000 per month during the school year, save up $6500 this summer, or (in the middle) save $2500 and earn $500/month during the school year. This is all according to my pessimistic cost forecasts, so this is a worst case analysis. 

It does, however, use OSAP and UTAPS estimates that are not guaranteed. I will have to update these with actual figures once I've applied. Moreover, it treats OSAP loans as if they were revenue - loans are liabilities.