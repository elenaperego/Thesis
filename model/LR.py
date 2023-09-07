import numpy as np
import analysis.data_preprocessing
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# Auxiliary function to standardize a dataset
def standardize(data):
    mean_per_column = data.mean(axis=0, numeric_only=True)
    std_per_column = data.std(axis=0, numeric_only=True)
    return (data - mean_per_column)/std_per_column

# This function prints the regression coefficients with teh respective p-values given the data and the fitted model.
def compute_metrics(fit, X, Y):
    coeffs = fit.coef_
    std_errs = np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X)) * (np.sum((Y - np.dot(X, coeffs)) ** 2) / (X.shape[0] - X.shape[1] - 1))))
    t_values = coeffs / std_errs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), X.shape[0] - X.shape[1] - 1))
    print('Coefficients:', coeffs)
    print('P-values:', p_values)

# This function runs the Multivariate Linear Regression model and prints the results
def run_linear_regression(data_model):
    X = sm.add_constant(data_model)
    X = X.drop(['VADER_SCORE'], axis=1)
    Y = data_model['VADER_SCORE']
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())
    predicted_values = results.predict(X)
    residuals = Y - predicted_values
    print(np.mean(residuals ** 2))
    print('')

# SEPARATE SES
# Select data
data = analysis.data_preprocessing.get_full_data()
data_model = data.drop(['GEO_ID','NAME','AMERICAN_INDIAN_ALASKA_NATIVE','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER','SERVICE',
                  'SALES AND OFFICE','NATURAL RESOURCES, CONSTRUCTION, MAINTENANCE','PRODUCTION, TRANSPORTATION, MATERIAL MOVING',
                  'LESS_HIGH_SCHOOL','HIGH_SCHOOL_GRADUATE','ASIAN','HISPANIC_LATINO',
                  'AFRICAN_AMERICAN','WHITE','MANAGEMENT, BUSINESS, SCIENCE, ARTS','TREND'], axis = 1).dropna()

# Standardize
data_model[['UNEMPLOYMENT_RATE', 'PERCENTAGE_INSURED', 'RATIO_BLACK_TO_WHITE','DEGREE_HOLDERS','MEDIAN_HOUSEHOLD_INCOME',
            'TOTAL_POPULATION','MEDIAN_AGE','MALE']] = standardize(data_model[['UNEMPLOYMENT_RATE', 'PERCENTAGE_INSURED',
                                                                               'RATIO_BLACK_TO_WHITE','DEGREE_HOLDERS',
                                                                               'MEDIAN_HOUSEHOLD_INCOME','TOTAL_POPULATION',
                                                                               'MEDIAN_AGE','MALE']])

# Log
data_model['CASES'] = np.log10(data_model['CASES'])

# Run model
run_linear_regression(data_model)

# NSES
data = analysis.data_preprocessing.get_full_data_NSES()
data_model_NSES = data.drop(['GEO_ID','NAME','AMERICAN_INDIAN_ALASKA_NATIVE','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER','ASIAN','HISPANIC_LATINO',
                  'AFRICAN_AMERICAN','WHITE','RATIO_BLACK_TO_WHITE','TREND'], axis = 1).dropna()

# Standardize
data_model_NSES[['NSES','MEDIAN_AGE','MALE','TOTAL_POPULATION']] = standardize(data_model_NSES[['NSES','MEDIAN_AGE','MALE','TOTAL_POPULATION']])

# Log
data_model_NSES['CASES'] = np.log10(data_model_NSES['CASES'])

# Run model
run_linear_regression(data_model_NSES)





