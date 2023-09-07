# Thesis

This repository contains the code and resources for my thesis project.

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

