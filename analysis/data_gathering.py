import pandas as pd

path = 'C:\\Users\\lucac\\Desktop\\Thesis\\data'
cities = ["Atlanta city, Georgia","Baltimore city, Maryland","Boston city, Massachusetts","Charlotte city, North Carolina",
          "Chicago city, Illinois", "Cleveland city, Ohio", "Denver city, Colorado","Detroit city, Michigan",
          "Houston city, Texas", "Indianapolis city (balance), Indiana", "Las Vegas city, Nevada",
          "Los Angeles city, California", "Miami city, Florida", "Nashville-Davidson metropolitan government (balance), Tennessee",
          "New Orleans city, Louisiana", "New York city, New York", "Philadelphia city, Pennsylvania",
          "San Francisco city, California", "Seattle city, Washington", "Washington city, District of Columbia"]

def percent_to_float(percent_string):
    return float(percent_string.strip('%')) / 100

# Sentiment data
def get_sentiment_data():
    atlanta = pd.read_csv(path+'\\urban_rural\\data\\Atlanta_GA\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    baltimore = pd.read_csv(path+'\\urban_rural\\data\\Baltimore_MD\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    boston = pd.read_csv(path+'\\urban_rural\\data\\Boston_MA\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    charlotte = pd.read_csv(path+'\\urban_rural\\data\\Charlotte_NC\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    chicago = pd.read_csv(path+'\\urban_rural\\data\\Chicago_IL\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    cleveland = pd.read_csv(path+'\\urban_rural\\data\\Cleveland_OH\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    denver = pd.read_csv(path+'\\urban_rural\\data\\Denver_CO\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    detroit = pd.read_csv(path+'\\urban_rural\\data\\Detroit_MI\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    houston = pd.read_csv(path+'\\urban_rural\\data\\Houston_TX\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    indianapolis = pd.read_csv(path+'\\urban_rural\\data\\Indianapolis_IN\\VADER_nonzero_daily_stats.tsv', sep="\t",
                               index_col=[0]).reset_index()
    lasvegas = pd.read_csv(path+'\\urban_rural\\data\\Las_Vegas_NV\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    losangeles = pd.read_csv(path+'\\urban_rural\\data\\Los_Angeles_CA\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    miami = pd.read_csv(path+'\\urban_rural\\data\\Miami_FL\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    nashville = pd.read_csv(path+'\\urban_rural\\data\\Nashville_TN\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    neworleans = pd.read_csv(path+'\\urban_rural\\data\\New_Orleans_LA\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    newyork = pd.read_csv(path+'\\urban_rural\\data\\New_York_NY\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    philadelphia = pd.read_csv(path+'\\urban_rural\\data\\Philadelphia_PA\\VADER_nonzero_daily_stats.tsv', sep="\t",
                               index_col=[0]).reset_index()
    sanfrancisco = pd.read_csv(path+'\\urban_rural\\data\\San_Francisco_CA\\VADER_nonzero_daily_stats.tsv', sep="\t",
                               index_col=[0]).reset_index()
    seattle = pd.read_csv(path+'\\urban_rural\\data\\Seattle_WA\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    washington = pd.read_csv(path+'\\urban_rural\\data\\Washington_DC\\VADER_nonzero_daily_stats.tsv', sep="\t", index_col=[0]).reset_index()
    return atlanta, baltimore, boston, charlotte, chicago, cleveland, denver, detroit, houston, indianapolis, lasvegas, \
           losangeles, miami, nashville, neworleans, newyork, philadelphia, sanfrancisco, seattle, washington


# COVID19 data: https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/jhu-csse.html
def get_covid_data():
    # covid_data = covidcast.signal('jhu-csse', 'confirmed_7dav_incidence_prop', date(2020, 2, 20), date(2020, 7, 31),
    #                              geo_type='county')
    # The data has been saved to a csv file to speed up the data gathering processs
    covid_data = pd.read_csv(path+'\\covid_data.csv',dtype=str)
    FIPS_to_county = pd.read_csv(path+'\\urban_rural\\data\\urban_counties.tsv', sep='\t', dtype=str)
    covid_data['geo_value'] = covid_data['geo_value'].astype(str)
    final = pd.merge(FIPS_to_county[['FIPS', 'city']], covid_data, left_on='FIPS', right_on='geo_value', how='inner')
    return final


# SES
# median income https://data.census.gov/table?q=median+household+income+in+cities+in+2020&tid=ACSST5Y2020.S1901
def get_median_household_income():
    median_income = pd.read_csv(path+'\\SES\\median_income.csv', usecols=['GEO_ID', 'NAME', 'S1901_C01_012E'], low_memory=False)
    #median_income.drop(index=median_income.index[0], axis=0, inplace=True)
    median_income = median_income[median_income['NAME'].isin(cities)]
    median_income = median_income.rename(columns={'S1901_C01_012E': 'MEDIAN_HOUSEHOLD_INCOME'})
    median_income['MEDIAN_HOUSEHOLD_INCOME'] = median_income['MEDIAN_HOUSEHOLD_INCOME'].astype(float)
    return median_income

# educational attainment https://data.census.gov/table?q=education+attainment+2020&g=010XX00US$1600000&tid=ACSST5Y2020.S1501
def get_education_attainment():
    education = pd.read_csv(path+'\\SES\\education_attainment.csv', usecols=['GEO_ID', 'NAME', 'S1501_C01_002E', 'S1501_C01_007E','S1501_C01_008E', 'S1501_C01_003E', 'S1501_C01_009E', 'S1501_C01_014E', 'S1501_C01_017E', 'S1501_C01_020E', 'S1501_C01_023E', 'S1501_C01_026E', 'S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_013E','S1501_C01_015E', 'S1501_C01_018E', 'S1501_C01_021E', 'S1501_C01_024E', 'S1501_C01_027E'], low_memory=False)
    education = education[education['NAME'].isin(cities)]
    education[['S1501_C01_002E', 'S1501_C01_007E','S1501_C01_008E', 'S1501_C01_003E', 'S1501_C01_009E', 'S1501_C01_014E', 'S1501_C01_017E', 'S1501_C01_020E', 'S1501_C01_023E', 'S1501_C01_026E', 'S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_013E','S1501_C01_015E', 'S1501_C01_018E', 'S1501_C01_021E', 'S1501_C01_024E', 'S1501_C01_027E']] = education[['S1501_C01_002E', 'S1501_C01_007E','S1501_C01_008E', 'S1501_C01_003E', 'S1501_C01_009E', 'S1501_C01_014E', 'S1501_C01_017E', 'S1501_C01_020E', 'S1501_C01_023E', 'S1501_C01_026E', 'S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_013E','S1501_C01_015E', 'S1501_C01_018E', 'S1501_C01_021E', 'S1501_C01_024E', 'S1501_C01_027E']].astype(float)
    education['DEGREE_HOLDERS'] = education['S1501_C01_005E'] + education['S1501_C01_012E'] + education['S1501_C01_015E']+ education['S1501_C01_018E']+ education['S1501_C01_021E']+ education['S1501_C01_024E']+ education['S1501_C01_027E']
    education['LESS_HIGH_SCHOOL'] = education['S1501_C01_002E'] + education['S1501_C01_007E'] + education['S1501_C01_008E']
    education['HIGH_SCHOOL_GRADUATE'] = education['S1501_C01_003E']+education['S1501_C01_009E'] + education['S1501_C01_014E'] + education['S1501_C01_017E'] + education['S1501_C01_020E']+ education['S1501_C01_023E']+ education['S1501_C01_026E']
    total = education['DEGREE_HOLDERS']+education['LESS_HIGH_SCHOOL']+education['HIGH_SCHOOL_GRADUATE']
    education['DEGREE_HOLDERS'] = education['DEGREE_HOLDERS']/total
    education['LESS_HIGH_SCHOOL'] = education['LESS_HIGH_SCHOOL']/total
    education['HIGH_SCHOOL_GRADUATE'] = education['HIGH_SCHOOL_GRADUATE']/total
    education = education[['GEO_ID', 'NAME', 'DEGREE_HOLDERS','LESS_HIGH_SCHOOL','HIGH_SCHOOL_GRADUATE']]
    return education

# only degree holders
def get_degree_holders():
    education = pd.read_csv(path+'\\SES\\education_attainment.csv', usecols=['GEO_ID', 'NAME', 'S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_015E','S1501_C01_018E','S1501_C01_021E','S1501_C01_024E','S1501_C01_027E'], low_memory=False)
    education = education[education['NAME'].isin(cities)]
    education[['S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_015E','S1501_C01_018E','S1501_C01_021E','S1501_C01_024E','S1501_C01_027E']] = education[['S1501_C01_005E', 'S1501_C01_012E', 'S1501_C01_015E','S1501_C01_018E','S1501_C01_021E','S1501_C01_024E','S1501_C01_027E']].astype(float)
    education['DEGREE_HOLDERS'] = education['S1501_C01_005E'] + education['S1501_C01_012E'] + education['S1501_C01_015E']+ education['S1501_C01_018E']+ education['S1501_C01_021E']+ education['S1501_C01_024E']+ education['S1501_C01_027E']
    education = education[['GEO_ID', 'NAME', 'DEGREE_HOLDERS']]
    return education


# occupation https://data.census.gov/table?q=unemployment+rate&g=010XX00US$1600000&y=2020
# unemployment rate https://data.census.gov/table?q=occupation+2020&g=010XX00US$1600000
def get_occupation():
    occupation = pd.read_csv(path+'\\SES\\occupation.csv',
                            usecols=['GEO_ID', 'NAME', 'S2406_C01_001E', 'S2406_C01_002E', 'S2406_C01_003E',
                                     'S2406_C01_004E', 'S2406_C01_005E', 'S2406_C01_006E'], low_memory=False)
    employment = pd.read_csv('C:\\Users\\lucac\\Desktop\\Thesis\\data\\SES\\employment.csv')
    employment = employment.iloc[[0]]
    employment = employment.T.reset_index()
    employment.columns = ['NAME', 'UNEMPLOYMENT_RATE']
    employment = employment.drop(index=0).reset_index(drop=True)
    employment['NAME'] = employment['NAME'].loc[
    employment['NAME'].str.contains('Estimate') & employment['NAME'].str.contains('Unemployment rate')]
    employment = employment.dropna()
    employment['NAME'] = employment['NAME'].str.split('!', n=1).str[0]
    employment['UNEMPLOYMENT_RATE'] = employment['UNEMPLOYMENT_RATE'].apply(percent_to_float)
    occupation = occupation[occupation['NAME'].isin(cities)]
    occupation = occupation.rename(columns={'S2406_C01_001E': 'TOTAL', 'S2406_C01_002E': 'MANAGEMENT, BUSINESS, SCIENCE, ARTS',
                                            'S2406_C01_003E': 'SERVICE', 'S2406_C01_004E': 'SALES AND OFFICE',
                                            'S2406_C01_005E': 'NATURAL RESOURCES, CONSTRUCTION, MAINTENANCE',
                                            'S2406_C01_006E': 'PRODUCTION, TRANSPORTATION, MATERIAL MOVING'})
    occupation['MANAGEMENT, BUSINESS, SCIENCE, ARTS'] = occupation['MANAGEMENT, BUSINESS, SCIENCE, ARTS'].astype(float)/occupation['TOTAL'].astype(float)
    occupation['SERVICE'] = occupation['SERVICE'].astype(float)/occupation['TOTAL'].astype(float)
    occupation['SALES AND OFFICE'] = occupation['SALES AND OFFICE'].astype(float)/occupation['TOTAL'].astype(float)
    occupation['NATURAL RESOURCES, CONSTRUCTION, MAINTENANCE'] = occupation['NATURAL RESOURCES, CONSTRUCTION, MAINTENANCE'].astype(float)/occupation['TOTAL'].astype(float)
    occupation['PRODUCTION, TRANSPORTATION, MATERIAL MOVING'] = occupation['PRODUCTION, TRANSPORTATION, MATERIAL MOVING'].astype(float)/occupation['TOTAL'].astype(float)
    occupation = occupation.drop('TOTAL', axis=1)
    return pd.merge(occupation, employment, on='NAME')

# percentage of the insured population https://data.census.gov/table?q=health+insurance+2020&g=010XX00US$1600000&tid=ACSST5Y2020.S2701
def get_health_coverage():
    health_coverage = pd.read_csv(path+'\\SES\\health_coverage.csv',
                            usecols=['GEO_ID', 'NAME', 'S2701_C01_001E', 'S2701_C02_001E', 'S2701_C04_001E'], low_memory=False)
    health_coverage = health_coverage[health_coverage['NAME'].isin(cities)]
    health_coverage = health_coverage.rename(columns={'S2701_C01_001E': 'TOTAL', 'S2701_C02_001E': 'INSURED',
                                     'S2701_C04_001E': 'UNINSURED'})
    health_coverage['PERCENTAGE_INSURED'] = health_coverage['INSURED'].astype(float)/health_coverage['TOTAL'].astype(float)
    health_coverage['PERCENTAGE_UNINSURED'] = health_coverage['UNINSURED'].astype(float)/health_coverage['TOTAL'].astype(float)
    return health_coverage

# demographics and race https://data.census.gov/table?q=population+2020&g=010XX00US$1600000&tid=ACSDP5Y2020.DP05&moe=false
def get_population_age_sex_race():
    population_age_sex_race = pd.read_csv(path+'\\SES\\populatio_sex_age_race.csv',
                                  usecols=['GEO_ID', 'NAME', 'DP05_0001E', 'DP05_0002E', 'DP05_0018E', 'DP05_0037E',
                                           'DP05_0038E', 'DP05_0039E', 'DP05_0044E', 'DP05_0052E', 'DP05_0071E'], low_memory=False)
    population_age_sex_race = population_age_sex_race[population_age_sex_race['NAME'].isin(cities)]
    population_age_sex_race = population_age_sex_race.rename(columns={'DP05_0001E': 'TOTAL_POPULATION', 'DP05_0002E': 'MALE',
                                                      'DP05_0018E': 'MEDIAN_AGE', 'DP05_0037E': 'WHITE', 'DP05_0038E': 'AFRICAN_AMERICAN',
                                                      'DP05_0039E': 'AMERICAN_INDIAN_ALASKA_NATIVE', 'DP05_0044E': 'ASIAN',
                                                      'DP05_0052E': 'NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER',
                                                      'DP05_0071E': 'HISPANIC_LATINO'})
    population_age_sex_race[['TOTAL_POPULATION','MEDIAN_AGE','MALE','WHITE', 'AFRICAN_AMERICAN','AMERICAN_INDIAN_ALASKA_NATIVE', 'ASIAN','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER','HISPANIC_LATINO']] = population_age_sex_race[['TOTAL_POPULATION','MEDIAN_AGE','MALE','WHITE', 'AFRICAN_AMERICAN','AMERICAN_INDIAN_ALASKA_NATIVE', 'ASIAN','NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER','HISPANIC_LATINO']].astype(float)
    total = population_age_sex_race['WHITE']+population_age_sex_race['AFRICAN_AMERICAN']+population_age_sex_race['AMERICAN_INDIAN_ALASKA_NATIVE']+population_age_sex_race['NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER']+population_age_sex_race['HISPANIC_LATINO']+population_age_sex_race['ASIAN']
    population_age_sex_race['MALE'] = population_age_sex_race['MALE']/population_age_sex_race['TOTAL_POPULATION']
    population_age_sex_race['WHITE'] = population_age_sex_race['WHITE']/total
    population_age_sex_race['AFRICAN_AMERICAN'] = population_age_sex_race['AFRICAN_AMERICAN']/total
    population_age_sex_race['AMERICAN_INDIAN_ALASKA_NATIVE'] = population_age_sex_race['AMERICAN_INDIAN_ALASKA_NATIVE']/total
    population_age_sex_race['ASIAN'] = population_age_sex_race['ASIAN']/total
    population_age_sex_race['NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER'] = population_age_sex_race['NATIVE_HAWAIIAN_OTHER_PACIFIC_ISLANDER']/total
    population_age_sex_race['HISPANIC_LATINO'] = population_age_sex_race['HISPANIC_LATINO']/total
    return population_age_sex_race