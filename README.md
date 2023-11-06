# Thesis

This repository contains the code and resources for my thesis project.
The COVID-19 pandemic has served as a global wake-up call, highlighting the urgent need to address gender, racial, and economic inequalities. This research paper explores the application of machine learning techniques in analyzing the impact of Socio-Economic Status (SES) on mental health in the United States during the COVID-19 pandemic using Twitter data. SES is defined in two ways: as separate variables and as combined variables using the Neighborhood Socio-economic Status index developed by the U.S. Census. This index is defined as 𝑁𝑆𝐸𝑆=log(𝑚𝑒𝑑𝑖𝑎𝑛 ℎ𝑜𝑢𝑠𝑒ℎ𝑜𝑙𝑑 𝑖𝑛𝑐𝑜𝑚𝑒)+(−1.129×log(% 𝑜𝑓 𝑓𝑒𝑚𝑎𝑙𝑒 ℎ𝑒𝑎𝑒𝑑 ℎ𝑜𝑢𝑠𝑒ℎ𝑜𝑙𝑑𝑠))+(−1.104×log(𝑢𝑛𝑒𝑚𝑝𝑙𝑜𝑦𝑚𝑒𝑛𝑡 𝑟𝑎𝑡𝑒))+(−1.974×(% 𝑜𝑓 𝑝𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛 𝑏𝑒𝑙𝑜𝑤 𝑝𝑜𝑣𝑒𝑟𝑡𝑦))+(0.451×(% 𝑜𝑓 ℎ𝑖𝑔ℎ 𝑠𝑐ℎ𝑜𝑜𝑙 𝑔𝑟𝑎𝑑𝑢𝑎𝑡𝑒𝑠+2×(% 𝑜𝑓 𝑏𝑎𝑐ℎ𝑒𝑙𝑜𝑟′𝑠 𝑑𝑒𝑔𝑟𝑒𝑒 ℎ𝑜𝑙𝑑𝑒𝑟𝑠))). Consequently, two machine learning approaches are investigated, Multivariate Linear Regression, which represents the current state-of-the-art, and Principal Component Regression, designed to address the multicollinearity issue present in the dataset. Both models are thoroughly tested and compared based on the two definitions of SES.

### Analysis folder
This folder generates the complete datasets used for analysis and modelling and performs Exploratory Data Analysis (EDA)
to gain insightful information on the final dataset before modelling. The python files are presented in order 
starting from data collection and ending with the analyses.
- data_gathered.py is the python file which cleans the original datasets generating usable DataFrames already transforming 
categorical data into percentages.
- data_preprocessing.py performs the feature engineering techniques discussed in my thesis report and computes the 
NSES indexes.
- EDA.ipynb contains the general EDA conducted to explore the statistical properties of the dataset, among which the 
correlation between the variables and their relationship with each other. This analysis leads to the selection fo the 
final features of the implemented models.
- EDA_trend.ipynb contains the formulation of the new engineered feature called "trend" and the related experiments in 
determining its relevance for modelling purposes.


### Model folder
This folder contains the implementation fo the two models proposed in my thesis project, namely the Multivariate Linear 
Regression and the Principal Component Regression.
- LR.py is the implementation of the Multivariate Linear
Regression. This file can be run and its results are printed in the terminal.
- PCR.py is the implementation of the Principal Component
Regression. This file can be run and its results are printed in the terminal together with auxiliary visualization aimed 
to more clearly interpret the findings.
- LR_assumptions.ipynb visualizes with specific plots to assess if the final features meet the linear regression's 
assumptions used by both models.

