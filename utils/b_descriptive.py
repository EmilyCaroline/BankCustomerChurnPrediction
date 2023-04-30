import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.process_data import preporcess_data

plt.style.use('seaborn-muted')

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds = ds.iloc[:, 1:]  # remove IdClient

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

dataset1_csv_file = "data/Dataset2.csv"
dataset2_csv_file = "data/Credit Card Customer Churn.csv"
merged_df = preporcess_data(dataset1_csv_file, dataset2_csv_file)

# print(merged_df.info())


def customerGender():
    gender_counts = merged_df['Gender'].value_counts()
    return gender_counts


def activeMember():
    active_customers = merged_df['IsActiveMember'].value_counts()[1]
    return active_customers


def exitedCustomers():
    num_exited_customers = (merged_df['Exited'] == 1).sum()
    return num_exited_customers


def creditCardCustomer():
    total_customers = merged_df['Age'].value_counts().sum()
    num_customers_with_cc = merged_df['HasCrCard'].value_counts()[1]
    num_customers_without_cc = total_customers - num_customers_with_cc

    return total_customers, num_customers_with_cc, num_customers_without_cc


def piecharts(subtype: int = 0):
    categ = merged_df.select_dtypes(include='object').columns
    if subtype == 0:  # Percentage of Churn
        sns.set(font_scale=1.5)
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))

        labels = 'Churn', 'Remain'
        sizes = [merged_df.Exited[merged_df['Exited'] == 1].count(
        ), merged_df.Exited[merged_df['Exited'] == 0].count()]
        explode = (0, 0.1)
        axs.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        axs.axis('equal')
        plt.title("Ratio of customers churned", size=15)
    if subtype == 3:  # Percentage of Credit Card Customers
        sns.set(font_scale=1.5)
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))

        labels = 'Has Credit Card', 'Has No Credit Card'
        num_customers_with_cc_ratio = (
            (merged_df['HasCrCard'].value_counts()/len(merged_df))*100)[1]
        num_customers_without_cc_ratio = (
            (merged_df['HasCrCard'].value_counts()/len(merged_df))*100)[0]

        sizes = [num_customers_with_cc_ratio, num_customers_without_cc_ratio]
        explode = (0, 0.1)
        axs.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        axs.axis('equal')
        plt.title("Ratio of customers with credit cards", size=15)
    elif subtype == 1:  # social features
        sns.set(font_scale=.5)
        fig, axs = plt.subplots(2, 1, figsize=(20, 10))
        for j in list(range(2)):
            print(categ[j+1])
            axs[j].title.set_text(categ[j+1])
            axs[j].pie(merged_df[categ[j+1]].value_counts(normalize=True).values,
                       labels=merged_df[categ[j+1]].value_counts(normalize=True).index.values, autopct='%1.1f%%')
    return fig


def heatmap():
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(10, 10))         # Sample figsize in inches
    sns.heatmap(merged_df.corr(), vmin=0, vmax=1, annot=True,
                cmap='rainbow', ax=ax, fmt=".1f")
    fig.suptitle("Visualising correlation coefficient")
    return fig


def histos(numvarname: str = 'Age'):
    sns.set(font_scale=1)
    varlist = merged_df.select_dtypes(exclude='object').columns

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=.5)
    fig.suptitle(numvarname.replace('_', ' '))
    sns.histplot(ax=axs[0], x=merged_df[numvarname], kde=True)
    axs[0].title.set_text('Histogram (all samples)')
    sns.boxplot(ax=axs[1], y=numvarname, x='Exited', data=merged_df)
    axs[1].title.set_text('Boxplot by existed flag')
    axs[1].set_xlabel('')

    return fig, varlist


def scatters(varname1: str = 'Age', varname2: str = 'Balance'):
    sns.set(font_scale=1)
    varlist1 = merged_df.iloc[:, 1:].columns.to_list()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(varname2.replace('_', ' ')+' by '+varname1.replace('_', ' '))
    sns.scatterplot(ax=axs, x=varname1, y=varname2,
                    hue='Exited', data=merged_df)

    return fig, varlist1


def churnByCountry():
    sns.set(font_scale=1)
    plt.style.use('classic')
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    sns.countplot(x='Geography', hue='Exited', data=merged_df,
                  palette=['palegreen', 'darkseagreen'], ax=axs)
    axs.legend(['Not Churned', 'Churned'])
    axs.set_xlabel('Country', fontsize=14)
    axs.set_ylabel('Count', fontsize=14)
    fig.suptitle('Churn by Country', fontsize=16)

    return fig


def propChurnAndRemain():
    plt.style.use('seaborn')

    colors = ['#5f9ea0', '#ffb6c1']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    geography_counts = merged_df.groupby(
        ['Geography', 'Exited']).size().unstack(fill_value=0)
    geography_counts.plot(kind='bar', stacked=True,
                          ax=axes[0, 0], rot=0, color=colors)
    axes[0, 0].set_title('Counts of Geography by Customer Churn', color='blue')
    axes[0, 0].spines['bottom'].set_color('blue')
    axes[0, 0].spines['left'].set_color('blue')
    axes[0, 0].tick_params(axis='x', colors='blue')
    axes[0, 0].tick_params(axis='y', colors='blue')

    gender_counts = merged_df.groupby(
        ['Gender', 'Exited']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', stacked=True,
                       ax=axes[0, 1], rot=0, color=colors)
    axes[0, 1].set_title('Counts of Gender by Customer Churn', color='red')
    axes[0, 1].spines['bottom'].set_color('red')
    axes[0, 1].spines['left'].set_color('red')
    axes[0, 1].tick_params(axis='x', colors='red')
    axes[0, 1].tick_params(axis='y', colors='red')

    has_credit_counts = merged_df.groupby(
        ['HasCrCard', 'Exited']).size().unstack(fill_value=0)
    has_credit_counts.plot(kind='bar', stacked=True,
                           ax=axes[1, 0], rot=0, color=colors)
    axes[1, 0].set_title(
        'number of churning customers with an active card', color='#4B0082')
    axes[1, 0].spines['bottom'].set_color('#4B0082')
    axes[1, 0].spines['left'].set_color('#4B0082')
    axes[1, 0].tick_params(axis='x', colors='#4B0082')
    axes[1, 0].tick_params(axis='y', colors='#4B0082')

    active_member_counts = merged_df.groupby(
        ['IsActiveMember', 'Exited']).size().unstack(fill_value=0)
    active_member_counts.plot(kind='bar', stacked=True,
                              ax=axes[1, 1], rot=0, color=colors)
    axes[1, 1].set_title(
        'Customer Churn Figures for Active Members', color='green')
    axes[1, 1].spines['bottom'].set_color('green')
    axes[1, 1].spines['left'].set_color('green')
    axes[1, 1].tick_params(axis='x', colors='green')
    axes[1, 1].tick_params(axis='y', colors='green')
    fig.suptitle('Proportion of churn and Remaining customer', fontsize=16)

    fig.tight_layout()

    return fig


def outlierWithBoxplot():
    plt.style.use('seaborn')
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 14))

    sns.boxplot(x='Exited', y='Age', data=merged_df,
                ax=axs[0, 0], color='rosybrown')
    sns.boxplot(x='Exited', y='CreditScore', data=merged_df,
                ax=axs[0, 1], color='lightsalmon')
    sns.boxplot(x='Exited', y='Balance', data=merged_df,
                ax=axs[1, 0], color='powderblue')
    sns.boxplot(x='Exited', y='Tenure', data=merged_df,
                ax=axs[1, 1], color='tan')
    sns.boxplot(x='Exited', y='NumOfProducts',
                data=merged_df, ax=axs[0, 2], color='red')
    sns.boxplot(x='Exited', y='EstimatedSalary',
                data=merged_df, ax=axs[1, 2], color='blue')
    sns.boxplot(x='Exited', y='Investment', data=merged_df,
                ax=axs[2, 0], color='green')
    sns.boxplot(x='Exited', y='Yearly Tax', data=merged_df,
                ax=axs[2, 1], color='yellow')
    sns.boxplot(x='Exited', y='Activity', data=merged_df,
                ax=axs[2, 2], color='white')

    axs[0, 0].set_xticklabels(['Not Churn', 'Churn'])
    axs[0, 1].set_xticklabels(['Not Churn', 'Churn'])
    axs[1, 0].set_xticklabels(['Not Churn', 'Churn'])
    axs[1, 1].set_xticklabels(['Not Churn', 'Churn'])
    axs[1, 1].set_xticklabels(['Not Churn', 'Churn'])
    axs[0, 2].set_xticklabels(['Not Churn', 'Churn'])
    axs[1, 2].set_xticklabels(['Not Churn', 'Churn'])
    axs[1, 2].set_xticklabels(['Not Churn', 'Churn'])
    axs[2, 0].set_xticklabels(['Not Churn', 'Churn'])
    axs[2, 1].set_xticklabels(['Not Churn', 'Churn'])
    axs[2, 2].set_xticklabels(['Not Churn', 'Churn'])

    axs[0, 0].set_xlabel('Customer Churn')
    axs[0, 0].set_ylabel('Age')
    axs[0, 0].set_title('Age by Customer Churn')

    axs[0, 1].set_xlabel('Customer Churn')
    axs[0, 1].set_ylabel('Credit Score')
    axs[0, 1].set_title('Credit Score by Customer Churn')

    axs[1, 0].set_xlabel('Customer Churn')
    axs[1, 0].set_ylabel('Balance')
    axs[1, 0].set_title('Balance by Customer Churn')

    axs[1, 1].set_xlabel('Customer Churn')
    axs[1, 1].set_ylabel('Tenure')
    axs[1, 1].set_title('Tenure by Customer Churn')

    axs[0, 2].set_xlabel('Customer Churn')
    axs[0, 2].set_ylabel('Number Of Products')
    axs[0, 2].set_title('Number Of Products by Customer Churn')

    axs[1, 2].set_xlabel('Customer Churn')
    axs[1, 2].set_ylabel('Estimated Salary')
    axs[1, 2].set_title('Estimated Salary by Customer Churn')

    axs[2, 0].set_xlabel('Customer Churn')
    axs[2, 0].set_ylabel('Investment')
    axs[2, 0].set_title('Investment by Customer Churn')

    axs[2, 1].set_xlabel('Customer Churn')
    axs[2, 1].set_ylabel('Yearly Tax')
    axs[2, 1].set_title('Yearly Tax by Customer Churn')

    axs[2, 2].set_xlabel('Customer Churn')
    axs[2, 2].set_ylabel('Activity')
    axs[2, 2].set_title('Activity by Customer Churn')

    fig.suptitle('Visualising outlier with Boxplot', fontsize=16)

    fig.tight_layout()
    return fig


def propIncome():
    c = merged_df['EstimatedSalary'].value_counts(bins=5, sort=False)
    p = merged_df['EstimatedSalary'].value_counts(
        bins=5, sort=False, normalize=True)
    df = pd.concat([c, p], axis=1, keys=['Counts', '%'])

    return df
