import pandas as pd
import numpy as np


def preporcess_data(dataset1, dataset2):
    # Read Data Files
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)
    # remove special character
    df2.columns = df2.columns.str.replace(' ', '')

    # Remname the columns in df2 to match the column in df1
    df2 = df2.rename(columns={'CUST_ID': 'CustomerId', 'AGE': 'Age', 'INCOME': 'EstimatedSalary', 'CHURN': 'Exited',
                              'INVESTMENT': 'Investment', 'ACTIVITY': 'Activity', 'YRLY_AMT': 'Yearly Amt',
                              'AVG_DAILY_TX': 'Avg Daily Tax', 'AVG_DAILY_TX': 'Avg Daily Tax',
                              'YRLY_TX': 'Yearly Tax', 'AVG_TX_AMT': 'Avg Tax Amt', 'NEGTWEETS': 'Negtweets', 'STATE': 'State',
                              'EDUCATION': 'Education',
                              'EDUCATION_GROUP': 'Education Group', 'TWITTERID': 'TwitterID'
                              })

    # Since dataset is for all the different states in US, we can assume that the Geography value as 'United States'
    if 'Geography' not in df2:
        df2.insert(loc=1,
                   column='Geography',
                   value='United States')

    # Applying the condition to create a gender column based on the value in SEX column of the df2
    df2['Gender'] = np.where((df2['SEX'] == 'F'), 'Female', 'Male')

    # merged two data files
    merged_df = pd.merge(df1, df2, on=[
                         'CustomerId', 'Age', 'Gender', 'EstimatedSalary', 'Exited', 'Geography'], how='outer')

    # Drop Unnecessary Columns
    merged_df.drop(['RowNumber', 'CustomerId', 'Surname', 'SEX', 'CHURN_LABEL',
                   'State', 'Negtweets', 'TwitterID'], axis=1, inplace=True)

    # Drop missing values
    drop_row_all = merged_df.dropna(how='all')
    # missings = drop_row_all.isna().sum().sum()

    # Filling the missing data with the mean or median value if it’s a numerical variable.
    # Filling the missing data with mode if it’s a categorical value.
    # For state, which is only applicable for US and not applicable for other countries, we can consider to fill it with NA = Not Applicable

    merged_df['CreditScore'] = merged_df['CreditScore'].fillna(
        merged_df['CreditScore'].mean())
    merged_df['Balance'] = merged_df['Balance'].fillna(
        merged_df['Balance'].mean())
    merged_df['NumOfProducts'] = merged_df['NumOfProducts'].fillna(
        merged_df['NumOfProducts'].mean())

    merged_df['Investment'] = merged_df['Investment'].fillna(
        merged_df['Investment'].mean())
    merged_df['Activity'] = merged_df['Activity'].fillna(
        merged_df['Activity'].mean())
    merged_df['Yearly Amt'] = merged_df['Yearly Amt'].fillna(
        merged_df['Yearly Amt'].mean())
    merged_df['Avg Daily Tax'] = merged_df['Avg Daily Tax'].fillna(
        merged_df['Avg Daily Tax'].mean())
    merged_df['Yearly Tax'] = merged_df['Yearly Tax'].fillna(
        merged_df['Yearly Tax'].mean())
    merged_df['Avg Tax Amt'] = merged_df['Avg Tax Amt'].fillna(
        merged_df['Avg Tax Amt'].mean())
    merged_df['Education'] = merged_df['Education'].fillna(
        merged_df['Education'].mean())

    merged_df['Education Group'] = merged_df['Education Group'].fillna(
        merged_df['Education Group'].mode()[0])

    merged_df['Tenure'] = merged_df['Tenure'].fillna(
        merged_df['Tenure'].mean())

    from sklearn import impute
    im = impute.SimpleImputer(missing_values=np.nan, strategy='median')

    merged_df['HasCrCard'] = im.fit_transform(merged_df[['HasCrCard']])
    merged_df['IsActiveMember'] = im.fit_transform(
        merged_df[['IsActiveMember']])

    return merged_df
