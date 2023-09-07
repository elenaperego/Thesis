import numpy as np
from scipy.stats import linregress
import analysis.data_gathering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

cities = ["Atlanta city, Georgia", "Baltimore city, Maryland", "Boston city, Massachusetts", "Charlotte city, North Carolina",
          "Chicago city, Illinois", "Cleveland city, Ohio", "Denver city, Colorado", "Detroit city, Michigan",
          "Houston city, Texas", "Indianapolis city (balance), Indiana", "Las Vegas city, Nevada",
          "Los Angeles city, California", "Miami city, Florida",
          "Nashville-Davidson metropolitan government (balance), Tennessee",
          "New Orleans city, Louisiana", "New York city, New York", "Philadelphia city, Pennsylvania",
          "San Francisco city, California", "Seattle city, Washington", "Washington city, District of Columbia"]

index = ['Atlanta, GA', 'Baltimore, MD', 'Boston, MA', 'Charlotte, NC', 'Chicago, IL', 'Cleveland, OH', 'Denver, CO',
         'Detroit, MI', 'Houston, TX', 'Indianapolis, IN', 'Las Vegas, NV', 'Los Angeles, CA', 'Miami, FL', 'Nashville, TN',
         'New Orleans, LA', 'New York, NY', 'Philadelphia, PA', 'San Francisco, CA', 'Seattle, WA', 'Washington, DC']

matching_dict = dict(zip(index, cities))

# Auxiliary function to normalize a dataset
def normalize(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

# Create dataset containing all SES
def get_SES():
    median_income = analysis.data_gathering.get_median_household_income()
    education = analysis.data_gathering.get_education_attainment()
    occupation = analysis.data_gathering.get_occupation()
    health_coverage = analysis.data_gathering.get_health_coverage()
    merge1 = pd.merge(median_income, education, on=['GEO_ID', 'NAME'], how='outer')
    merge2 = pd.merge(merge1, occupation, on=['GEO_ID', 'NAME'], how='outer')
    SES = pd.merge(merge2, health_coverage[['GEO_ID', 'NAME', 'PERCENTAGE_INSURED']], on=['GEO_ID', 'NAME'],
                      how='outer')
    return SES

# Create dataset containing all SES
def get_demographics():
    population_age_sex_race = analysis.data_gathering.get_population_age_sex_race()
    return population_age_sex_race

# Create dataset the VADER scores averaged per city
def get_mean_VADER():
    sentiment_compressed = []
    sentiment_compressed.extend(analysis.data_gathering.get_sentiment_data())
    sentiment = pd.DataFrame(columns=['NAME', 'VADER_SCORE'])
    for i in range(0, len(cities)):
        city = sentiment_compressed[i]
        sentiment.loc[i] = [cities[i], city['mean'].mean()]
    return sentiment

# Create dataset containing number of COVID-19 cases averaged per city
def get_mean_COVID_cases():
    covid_data = analysis.data_gathering.get_covid_data()
    covid_data['value'] = covid_data['value'].astype(float)
    covid_data_grouped = covid_data.groupby('city')['value'].mean().reset_index()
    covid_data_grouped['city'] = covid_data_grouped['city'].map(matching_dict)
    covid_data_grouped = covid_data_grouped.rename(columns={'city': 'NAME', 'value': 'CASES'})
    return covid_data_grouped

def percent_to_float(percent_string):
    return float(percent_string.strip('%')) / 100

# NSES index: percentage of population below poverty level
# https://data.census.gov/table?q=percent+below+poverty&g=160XX00US0644000,0667000,0820000,1150000,1245000,1304000,1714000,1836003,2255000,2404000,2507000,2622000,3240000,3651000,3712000,3916000,4260000,4752006,4835000,5363000&y=2020
def get_below_poverty():
    below_poverty = pd.read_csv('C:\\Users\\lucac\\Desktop\\Thesis\\data\\SES\\below_poverty.csv')
    below_poverty = below_poverty.iloc[[0]]
    below_poverty = below_poverty.T.reset_index()
    below_poverty.columns = ['NAME', 'PERCENTAGE_BELOW_POVERTY']
    below_poverty = below_poverty.drop(index=0).reset_index(drop=True)
    below_poverty['NAME'] = below_poverty['NAME'].loc[
    below_poverty['NAME'].str.contains('Estimate') & below_poverty['NAME'].str.contains('Percent below poverty level')]
    below_poverty = below_poverty.dropna()
    below_poverty['NAME'] = below_poverty['NAME'].str.split('!', n=1).str[0]
    below_poverty['PERCENTAGE_BELOW_POVERTY'] = below_poverty['PERCENTAGE_BELOW_POVERTY'].apply(percent_to_float)
    return below_poverty

# NSES index: percentage of female-headed households
# https://data.census.gov/table?q=DP02&g=160XX00US0644000,0667000,0820000,1150000,1245000,1304000,1714000,1836003,2255000,2404000,2507000,2622000,3240000,3651000,3712000,3916000,4260000,4752006,4835000,5363000&y=2020&tid=ACSDP5Y2020.DP02&moe=false
def get_female_household():
    female_household = pd.read_csv('C:\\Users\\lucac\\Desktop\\Thesis\\data\\SES\\female_household.csv')
    female_household = female_household.T.reset_index()
    female_household = female_household.iloc[:, [0, 11, 13]]
    female_household = female_household.drop(index=0).reset_index(drop=True)
    female_household.columns = ['NAME', 'TOTAL', 'SINGLE']
    female_household['NAME'] = female_household['NAME'].loc[female_household['NAME'].str.contains('Percent')]
    female_household = female_household.dropna()
    female_household['NAME'] = female_household['NAME'].str.split('!', n=1).str[0]
    female_household['PERCENTAGE_FEMALE_HOUSEHOLD'] = female_household['TOTAL'].apply(percent_to_float)-female_household['SINGLE'].apply(percent_to_float)
    female_household['PERCENTAGE_FEMALE_HOUSEHOLD'] = female_household['PERCENTAGE_FEMALE_HOUSEHOLD']
    return female_household[['NAME','PERCENTAGE_FEMALE_HOUSEHOLD']].reset_index(drop=True)

# NSES index: unemployment rate
def get_unemployment_rate():
    occupation = analysis.data_gathering.get_occupation()
    return occupation[['NAME','UNEMPLOYMENT_RATE']]

# NSES index: median household income
def get_median_household_income():
    income = analysis.data_gathering.get_median_household_income()
    return income

# NSES index: percentage of graduates
def get_number_graduate():
    education = analysis.data_gathering.get_education_attainment()
    return education[['NAME','HIGH_SCHOOL_GRADUATE']]

# NSES index: percentage of degree holders
def get_number_degree():
    education = analysis.data_gathering.get_education_attainment()
    return education[['NAME','DEGREE_HOLDERS']]

# NSES index: final calculation
# NSES = log(median household income) + (-1.129 * (log(percent of female-headed households))) + (-1.104 *
# (log(unemployment rate))) + (-1.974 * (log(percent below poverty))) + .451*((high school grads)+(2*(bachelor's degree
# holders)))
def get_NSES():
    merge1 = pd.merge(get_median_household_income(), get_female_household(), on='NAME')
    merge2 = pd.merge(merge1,get_unemployment_rate(),on='NAME')
    merge3 = pd.merge(merge2, get_below_poverty(), on='NAME')
    merge4 = pd.merge(merge3, get_number_graduate(), on='NAME')
    nses = pd.merge(merge4, get_number_degree(), on='NAME')
    nses['NSES'] = np.log(nses['MEDIAN_HOUSEHOLD_INCOME']) + (-1.129 * (np.log(nses['PERCENTAGE_FEMALE_HOUSEHOLD']))) + (-1.104 *
        (np.log(nses['UNEMPLOYMENT_RATE']))) + (-1.974 * (np.log(nses['PERCENTAGE_BELOW_POVERTY']))) + .451*((nses['HIGH_SCHOOL_GRADUATE'])
        +(2*(nses['DEGREE_HOLDERS'])))
    return nses[['NAME','NSES']]

# Trend: function to compute regression line's slope
def compute_regression_slope(group):
    slope, intercept, r_value, p_value, std_err = linregress(group['AVERAGE_CASES'],
                                                             group['WEIGHTED_AVERAGE_VADER'])
    return slope

# Trend: final trend measure between weekly time series
def get_trend_weekly():
    covid = analysis.data_gathering.get_covid_data()
    covid = covid.loc[:, ['FIPS', 'city', 'time_value', 'value']].dropna()
    covid_cities = dict(tuple(covid.groupby('city')))
    dfs = []
    for city in index:
        city_covid = covid_cities[city]
        sentiment_data = next(data for data in analysis.data_gathering.get_sentiment_data())
        merged_data = pd.merge(sentiment_data[['index', 'mean', 'counts']], city_covid, left_on='index',
                               right_on='time_value', how='inner')
        dfs.append(merged_data)
    result = pd.concat(dfs)
    result = result.reset_index(drop=True)
    result['city'] = result['city'].map(matching_dict)
    result = result.rename(columns={'mean': 'vader', 'value': 'cases'})
    result = result.drop('time_value', axis=1)
    result['index'] = pd.to_datetime(result['index'])
    result[['vader', 'counts', 'cases']] = result[['vader', 'counts', 'cases']].astype(float)
    weighted_avg_vader_city = result.groupby([pd.Grouper(key='index', freq='W'), 'city']).apply(
        lambda x: (x['vader'] * x['counts']).sum() / x['counts'].sum())
    avg_cases_city = result.groupby([pd.Grouper(key='index', freq='W'), 'city'])['cases'].mean()
    weekly_averages_city = pd.DataFrame(
        {'WEIGHTED_AVERAGE_VADER': weighted_avg_vader_city, 'AVERAGE_CASES': avg_cases_city}).reset_index()
    weekly_averages_city[['WEIGHTED_AVERAGE_VADER', 'AVERAGE_CASES']] = normalize(
        weekly_averages_city[['WEIGHTED_AVERAGE_VADER', 'AVERAGE_CASES']])
    grouped = weekly_averages_city.groupby('city')
    trends = pd.DataFrame({'trend': grouped.apply(compute_regression_slope)})
    trends.reset_index(inplace=True)
    trends = trends.rename(columns={'city': 'NAME', 'trend': 'TREND'})
    return trends

# Creates final full dataset using separate SES
def get_full_data():
    merge = pd.merge(get_SES(), get_mean_VADER(), on='NAME', how='outer')
    merge1 = pd.merge(merge, get_trend_weekly(), on='NAME', how='outer')
    merge2 = pd.merge(merge1, get_demographics(), on=['GEO_ID', 'NAME'], how='outer')
    data = pd.merge(merge2, get_mean_COVID_cases(), on='NAME', how='outer')
    data['RATIO_BLACK_TO_WHITE'] = data['AFRICAN_AMERICAN'] / data['WHITE']
    return data

# Creates final full dataset using NSES
def get_full_data_NSES():
    merge = pd.merge(get_NSES(), get_mean_VADER(), on='NAME', how='outer')
    merge1 = pd.merge(merge, get_trend_weekly(), on='NAME', how='outer')
    merge2 = pd.merge(merge1, get_demographics(), on='NAME', how='outer')
    data = pd.merge(merge2, get_mean_COVID_cases(), on='NAME', how='outer')
    data['RATIO_BLACK_TO_WHITE'] = data['AFRICAN_AMERICAN'] / data['WHITE']
    return data