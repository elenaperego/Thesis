import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import t
from sklearn.metrics import r2_score, mean_squared_error
import analysis.data_preprocessing
from sklearn.model_selection import KFold, cross_val_score
import math

# Auxiliary function to standardize a dataset
def standardize(data):
    mean_per_column = data.mean(axis=0, numeric_only=True)
    std_per_column = data.std(axis=0, numeric_only=True)
    return (data - mean_per_column)/std_per_column

# This function calculates the PCR's regression coefficients's p-values
def calculate_p_values(X, Y, model):
    n = X.shape[0]
    p = X.shape[1]
    df = n - p - 1
    resid_std = np.sqrt(np.sum((Y - model.predict(X)) ** 2) / df)
    XTX = np.dot(X.T, X)
    V = resid_std ** 2 * np.linalg.inv(XTX)
    se = np.sqrt(np.diag(V))
    t_stats = model.coef_ / se
    p_values = (1 - t.cdf(np.abs(t_stats), df)) * 2
    return p_values

# This function prints the PCR's results and visualizes them through a bar chart of the
# original features contributions to the target variable's prediction
def evaluation_metrics(linear_coef,components,p_values,n_components,X,Y,Y_predicted):
    # Evaluation metrics
    print('PCR Results:')
    print('============')
    print('R-squared: {:.4f}'.format(r2_score(Y, Y_predicted)))
    print('MSE: {:.4f}'.format(mean_squared_error(Y, Y_predicted)))
    print('\nPCR Coefficients:')
    for i in range(n_components):
        pc_num = i + 1
        pc_name = f'PC{pc_num}'
        print(f'{pc_name}:')
        pc_coef = linear_coef[i]
        pc_pval = p_values[i]
        print(f'  Coefficient: {pc_coef:.3f}, p-value: {pc_pval:.3f}')
        pc_features = components[i]
        for j, feat in enumerate(X.columns):
            print(f'  {feat}: {pc_features[j]:.3f}')
        print('\n')

        # Plotting original features contribution to the target variable for each principal component
        feature_contributions = pc_coef * pc_features
        sorted_indices = np.argsort(np.abs(feature_contributions))
        sorted_features = X.columns[sorted_indices]
        sorted_contributions = feature_contributions[sorted_indices]
        for j, feat in enumerate(sorted_features):
            print(f'  {feat}: {sorted_contributions[j]:.3f}')
        print('\n')
        colors = ['red' if val < 0 else 'green' for val in sorted_contributions]
        plt.figure(figsize=(18, 10))
        plt.barh(sorted_features, sorted_contributions, color=colors)
        for i, val in enumerate(sorted_contributions):
            plt.text(val, i, f'{val:.3f}', va='center', color='white' if val < 0 else 'black', fontsize=16)
        plt.xlabel('Contribution', fontsize=16)
        plt.ylabel('Original Features', fontsize=16)
        if math.floor(pc_pval*1000)/1000 <= 0.005:
            plt.title(f'Original Features Contribution to Target Variable of {pc_name}*', fontsize=22)
        if math.floor(pc_pval*1000)/1000 <= 0.001:
            plt.title(f'Original Features Contribution to Target Variable of {pc_name}***', fontsize=22)
        if math.floor(pc_pval*1000)/1000 <= 0.0001:
            plt.title(f'Original Features Contribution to Target Variable of {pc_name}****', fontsize=22)
        plt.axvline(x=0, color='black', linestyle='--')
        plt.gca().invert_yaxis()
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.savefig(f"features-pcr-{pc_name}.png", dpi=300,transparent=True)
        plt.show()

    # Variance Explained
    explained_variance = np.cumsum(fit_PCA.named_steps['pca'].explained_variance_ratio_)
    print('Variance Explained:')
    for i in range(n_components):
        pc_num = i + 1
        pc_name = f'PC{pc_num}'
        print(f'{pc_name}: {explained_variance[i]:.4f}')

# This function runs the PCR model while choosing the optimal number of principal components using Cross-Validation and
# visualizes this choice
def run_PCR(data_model):
    X = data_model.drop(['VADER'], axis=1)
    Y = data_model['VADER']
    X = standardize(X)

    # Selection of the optimal number of principal components using Cross-Validation
    n_components = min(X.shape[0], X.shape[1])
    pipeline = Pipeline(steps=[('pca', PCA()), ('linear', LinearRegression())])
    n_components_range = range(1, n_components + 1)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    for n in n_components_range:
        pipeline.named_steps['pca'].n_components = n
        scores = cross_val_score(pipeline, X, Y, scoring='neg_mean_squared_error', cv=cv)
        cv_scores.append(-np.mean(scores))

    best_n_components = n_components_range[np.argmin(cv_scores)]
    pipeline.named_steps['pca'].n_components = best_n_components
    fit_PCA = pipeline.fit(X, Y)
    Y_predicted_pcr = pipeline.predict(X)
    linear_coef = fit_PCA.named_steps['linear'].coef_
    pca_components = pipeline.named_steps['pca'].components_
    p_values = calculate_p_values(fit_PCA.named_steps['pca'].transform(X), Y, fit_PCA.named_steps['linear'])

    # Plotting Number of Components vs. Negative Mean Squared Error
    plt.figure(figsize=(12, 10))
    plt.plot(range(1, n_components + 1), cv_scores, marker='o')
    plt.xlabel('Number of Principal Components', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.title('Number of Components vs. Mean Squared Error', fontsize=22)
    best_n = np.argmin(cv_scores) + 1
    plt.axvline(x=best_n, color='red', linestyle='--', label=f'Best number of components ({best_n})')
    plt.legend()
    plt.savefig("components.png", dpi=300, transparent=True)
    plt.show()

    return linear_coef, pca_components, p_values, best_n_components, X, Y, Y_predicted_pcr, pipeline, fit_PCA


# SEPARATE SES
# Select data
# data = analysis.data_preprocessing.get_full_data()
# data_model = data.drop(['GEO_ID','NAME','AMERICAN_INDIAN_ALASKA_NATIVE','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER','SERVICE',
#                   'SALES AND OFFICE','NATURAL RESOURCES, CONSTRUCTION, MAINTENANCE','PRODUCTION, TRANSPORTATION, MATERIAL MOVING',
#                   'LESS_HIGH_SCHOOL','HIGH_SCHOOL_GRADUATE','ASIAN','HISPANIC_LATINO',
#                   'AFRICAN_AMERICAN','WHITE','MANAGEMENT, BUSINESS, SCIENCE, ARTS','TREND'], axis = 1).dropna()
#
# # Change labels
# labels_final = {'MEDIAN_HOUSEHOLD_INCOME':'income', 'DEGREE_HOLDERS':'degree holders', 'UNEMPLOYMENT_RATE':'unemployment',
#                 'PERCENTAGE_INSURED':'insurance','VADER_SCORE':'VADER','TOTAL_POPULATION':'population','MALE':'male',
#                 'MEDIAN_AGE':'age','CASES':'COVID-19 cases','RATIO_BLACK_TO_WHITE':'ratio black white','NSES':'NSES'}
# data_model = data_model.rename(columns=labels_final)
# linear_coef, pca_components, p_values, best_n_components, X, Y, Y_predicted_pcr, pipeline, fit_PCA = run_PCR(data_model)
#
# # Run model
# evaluation_metrics(linear_coef, pca_components, p_values, best_n_components, X, Y, Y_predicted_pcr)

# NSES
# Select data
data_NSES = analysis.data_preprocessing.get_full_data_NSES()
data_model_NSES = data_NSES.drop(['GEO_ID','NAME','TREND','AMERICAN_INDIAN_ALASKA_NATIVE','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER',
                             'ASIAN','HISPANIC_LATINO','AFRICAN_AMERICAN','WHITE','RATIO_BLACK_TO_WHITE'], axis = 1).dropna()

# Change labels
labels_final = {'VADER_SCORE':'VADER','TOTAL_POPULATION':'population','MALE':'male',
                'MEDIAN_AGE':'age','CASES':'COVID-19 cases','NSES':'NSES'}
data_model_NSES = data_model_NSES.rename(columns=labels_final)

# Run model
linear_coef, pca_components, p_values, best_n_components, X, Y, Y_predicted_pcr, pipeline, fit_PCA, = run_PCR(data_model_NSES)
evaluation_metrics(linear_coef, pca_components, p_values, best_n_components, X, Y, Y_predicted_pcr)







