<img src="https://lh4.googleusercontent.com/mJe6ZHK0AMSgXwm9f-eIrRXCRUxXoBcKpwAtYyxs3F74ceLFNW5LxIqhZvu5g69DyVpyvXA-xB26JWndQHdKu7dijpA8M9-EWF7laKKAm-Wbfum71eL9jo25FpQ2BnZqRH7-t0qPa1W2y1k39XO0sO4" alt="img" style="zoom:67%;" />

###                                                                     Asset Allocation Project (Group 7)



​                                 **WANG WANTING**

​                                 **LIU HUANSHA**

​                                 **LIU WEIJIA**

​                                 **WU HANYU**

​                                 **GUO LEIBING**

























## Introduction & Problem Statement

###### In an ever-evolving and volatile financial market, asset allocation stands as a pivotal strategy that balances the inherent trade-off between portfolio return and risk. The primary objective of this project is to provide valuable insights on shortlisting of robust asset allocation models that aligns the distbution of assets with the investment goals. We will delve into the past return data of  U.S 30 day TBill TR, Russell 2000 TR, S&P 500 TR, LB LT Gvt/Credit TR  and MSCI EAFE TR,  examining the statistics on different granularity (monthly, yearly) and proposing optimal strategies for the targeted time period and financial environment. We would expect the strategies to be tailored for different circumstances (e.g., high inflation, financial crisis etc) as the objectives and needs change accordingly.

###### The deliverable of the project will be presented on a dashboard built from a Pythonic framework known as Streamlit.



## Data Processing Logic:

The raw data are contained inside 3 separate excel sheets, each representing a different period (ie. 1980s, 1990s, 2000s). A simple workflow has been written in Python to concatenate all 3 sheets into 1 single DataFrame and then assigning a column called **Period** (3 distinct values: **'1980s'**, **'1990s'**, **'2000s'**) as a categorical indicator. Monthly return values are then converted from decimals to percentage by multiplying 100. The cleaned data will then be used for the analysis in the following questions.

We will only be using data points from 2000s to answer the questions below.



## Exploratory Data Analysis:

- S&P 500, Russell 2000 and MSCI EAFE have high positive correlation with each other
- All asset classes except T-bill have 100% distinct values
- No missing values in the data
- Time interval of the data is consistent

![Screenshot 2023-09-17 at 5.23.32 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.23.32 PM.png)















## Question 1: Calculating Return & Risk related statistics:

**The formula for calculating the statitics are as follow:**
$$
\begin{align*}
            &\text{Monthly Average Return} : 
            \frac{1}{n} \sum_{i=1}^{n} X_i \ \ \ \ \ \ 
            \text{Monthly Standard Deviation} : 
            \sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X})^2} \\
            &\text{Arithmetic Annual Return} : 12 
            \cdot \bar{X}\  \text{or}\ \frac{1}{N} \sum_{i=1}^{N} \left(\sum_{j=1}^{12} x_{ij}\right) \ \ \ \ \ \text{Annual Standard Deviation(1)}: 
            \sqrt{12} \times \sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X})^2}\\
            &\text{Geometric Annual Return} :\left(\prod _{i=1}^{n}(1+R_{i})\right)^{\frac {1}{n}}-1\\
            &\text{Annual Standard Deviation(2)} : \sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(\text{Annual Return}_i - \bar{Annual\ Return})^2}
            \end{align*}
$$
**Remark: ** 

Statistically, both arithmetic mean and geometric are unbiased estimators of average return **under different assumptions.** Arithmetic mean assumes the rate or return to be stable and consistent over time whereas the geometric mean do take in considerations of compounding effects and is alsohelpful when dealing with data with more fluctuations and unequal intervals. In the context of this question, arithmetic mean is preferred for estimating future average return because without knowing the true distribution of the future data, we would not want to overestimate or underestimate by considering the compounding effect. A modest approach is usually the safer bet.



(a) &(b)

The monthly and yearly statistics are shown in the table below, with maximum of each column colored in light coral. There are 2 approaches for calculating statistics at the annual level. One of the methods would be to calculate the monthly return and standard deviation first and then multiply by 12 and √12 to get the annual values.This is a rather **naive estimation** because it assumes that the monthly returns are **independent and identically distributed (i.i.d)** over time, which may not be the case in real world scenarios. The other approach would to **group by each individual year** and take the sum of return. Then we proceed to calculate the average and dispersion across the years. The annual return will be consistent for both methods but the standard deviation values will differ due to the i.i.d assumption. 

![Screenshot 2023-09-16 at 11.42.40 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-16 at 11.42.40 PM.png)



(c) The correlation matrix heatmap at both monthly and annual level are shown below.  In reality, correlation between assets can **change over time** due to various factors such as market condition, policies, market sentiments etc. Typically, we would not expect the monthly correlation to tally with the yearly correlation.

The correlation values seem to suggest that Russell 2000,  S&P 500 and MSCI EAFE tend to vary together and thus this reaffirms that their returns are not independent.

![Screenshot 2023-09-16 at 11.57.24 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-16 at 11.57.24 PM.png)



## Question 2: Optimal Asset Allocation

This is a constrained **(the range of values the weights can take are unbounded)** optimization problem. The objective is to derive a vector of weights that minimizes the portfolio standard deviation which is the square root of the product of the weights vector(1xn), the covariance matrix and the weights vector again (nx1). It is subjected to 2 constraints. First,  the expected return calculated from the dot product of weights and return must be equal to the target mean return, second, the weights must add up to 1.
$$
\begin{align*}
& \text{Objective Function:} \quad \mathbf{w}^* = \underset{\mathbf{w}}{\text{argmin}} \left(\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}\right) \\
& \text{Constraint 1:} \quad \text{Target Mean Return} - \text{Expected Return} = 0 \\
& \text{Constraint 2:} \quad \mathbf{w}^T \mathbf{1} = 1 \\
\end{align*}
$$
(a)

The optimization function is implemented both using the **trust region method** solver in the Python package Scipy(https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr) and Excel's non-linear GRG solver. The resulting weights and minimum standard deviation are tabulated below. The algorithm in Python achieved better results by producing weight combinations that yield smaller minimum standard deviation and higher Sharpe Ratio while converging towards target mean return.

<img src="/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.08.52 PM.png" alt="Screenshot 2023-09-17 at 5.08.52 PM" style="zoom: 67%;" />



The efficient frontier graph can be drawn as a scatter plot where the x coordinates are the minimum standard deviation values and the y coordinates are the expected mean return / target mean return. The capital market line can be plotted by setting the risk free rate (approximately 1.8) as the y-intercept and perform a linear regression that connects the  point on efficient frontier where it has the largest Sharpe Ratio (colored in light coral). The gradient ∇*f*(*x*) will be Δ*y*/Δ*x*.

![Screenshot 2023-09-17 at 5.11.46 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.11.46 PM.png)



(b) For the global minimum variance portfolio, the expected return coincides with the target mean return which is 8.2% and the dervied portfolio standard deviation is approximately **4.3**. The algorithm assigns **-0.084126** to **Russell 2000**, **0.045383** to **S&P 500**, **0.754658** to **LB LT Gvt/Credit** and **0.284084** to **MSCI EAFE**.

(c) The tangency portfolio has an expected return of 11.0%. The corresponding standard deviation is approximately 4.4. The algorithm assigns **0.210968** to **Russell 2000**, **-0.574201** to **S&P 500**, **0.822147** to **LB LT Gvt/Credit** and **0.541086** to **MSCI EAFE**.



## Question 3: Optimal Capital Allocation

(a) The dollar investment for each asset class from each investor can be calculated using the the weights of the tangency portfolio depending on the proportion of money investing into the risky assets **(0,2, 0.6 & 0.9 out of 1 million respectively)**

The tabulated results are shown below:

![Screenshot 2023-09-17 at 5.34.24 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.34.24 PM.png)

(b) The composition of the risky portfolio is the same since they take on the same weights, even though the absolute distribution of the wealth can be different. Based on the portfolio return and risk formula, we can calculate the portfolio return and risk of each investor normalized by their amount of weath devoted into risky assets. The relationship between portfolio risk and portfolio return is always **inversely proportional**. A conservative investor like Alex will be subjected to less risk but also less return. An investor with higher risk tolerance like Cathy will be able to make more money potentially while bearing more risk at the same time.
$$
\begin{align*}
&\text{Portfolio Return}: \R_p = R_f + \sum_{i=1}^{n} w_i \cdot (R_i - R_f) \\
&\text{Portfolio Risk}: \sigma_p = \sqrt{\sum \left[ w_i \cdot \sigma_i \right]^2 + 
            \sum\sum \left[ w_i \cdot w_j \cdot \sigma_i \cdot \sigma_j \cdot \rho_{ij} \right]}
\end{align*}
$$
![Screenshot 2023-09-17 at 5.52.34 PM](/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.52.34 PM.png)

(c) Our group has decided to go with a **high risk tolerance** of 100%. As such, all our wealth (assuming we have $1million as well) will be invested into risky assets. The following table shows the dollar invested in each asset class.

<img src="/Users/hanyuwu/Study/BMD5301/screenshots/Screenshot 2023-09-17 at 5.57.46 PM.png" alt="Screenshot 2023-09-17 at 5.57.46 PM" style="zoom: 50%;" />

