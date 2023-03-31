from utils.b_descriptive import piecharts, histos, scatters, creditCardCustomer, activeMember, exitedCustomers, customerGender, heatmap, churnByCountry, propChurnAndRemain, outlierWithBoxplot, propIncome
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon=":bank:",
    layout="wide"
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

plt.style.use("seaborn-v0_8")

st.image('./assets/uwe_logo.jpg')
st.title("Bank customers churn prediction")
st.sidebar.title("Bank customers churn prediction")
st.markdown("Welcome to this dashboard for bank customers churn insights! ")
st.sidebar.markdown(
    "Explore, predict churn rate of customers")

# 1st part of the dashboard
st.sidebar.title("Preliminary Data Analysis")
select1 = st.sidebar.selectbox('Select an option',
                               ['How many of the customer has credit card?', 'How many are active members?',
                                'How many customers have existed the bank?', 'What are the gender of customers?', "Propotion of different income level"], key='1')
if not st.sidebar.checkbox("Hide", True, key='2'):
    st.markdown("### 1. Preliminary Data Analysis")
    if select1 == 'How many of the customer has credit card?':
        total_customers = creditCardCustomer()[0]
        num_customers_with_cc = creditCardCustomer()[1]
        num_customers_without_cc = creditCardCustomer()[2]
        st.markdown("### How many of the customer has credit card?")
        st.write('Total Number of customers: ', total_customers)
        st.write("Number of customers with credit cards: ",
                 num_customers_with_cc)
        st.write("Number of customers without credit cards: ",
                 num_customers_without_cc)
        st.subheader(
            'Out of a total of 16,001 customers, 13,056 customers have credit cards. This means that approximately :green[81.6%] of the customers in the dataset have credit cards. The remaining :blue[18.4%] of the customers, or 2,945 customers, do not have credit cards.')

    if select1 == 'How many are active members?':
        st.markdown("### How many are active members?")
        active_members = activeMember()
        st.write('Number of active customers ', active_members)
        st.subheader(
            'Out of a total of 16,001 customers, 11,152 customers are active, while 4,849 customers are inactive. This means that approximately :green[69.7%] of the customers in the dataset are active, while the remaining :blue[30.3%] of the customers are inactive.')
    if select1 == 'How many customers have existed the bank?':
        st.markdown("### How many customers have existed the bank?")
        exited_customers = exitedCustomers()
        st.write('Number of exited customers: ', exited_customers)
        st.subheader(
            'Out of a total of 16,001 customers, 12,435 customers are still with the bank, while 3,566 customers have exited the bank. This means that approximately :green[77.7%] of the customers in the dataset are still with the bank, while the remaining :blue[22.3%] of the customers have exited.')
    if select1 == 'What are the gender of customers?':
        st.markdown("### What are the gender of customers")
        gender_count = customerGender()
        st.write('Male: ', gender_count[0])
        st.write('Female: ', gender_count[1])
        st.subheader(
            'Out of a total of 16,001 customers, 8,536 customers are male, while 7,465 customers are female. This means that approximately :green[53.3%] of the customers in the dataset are male, while the remaining :blue[46.7%] of the customers are female. This indicates that there is no significant gender imbalance in the dataset.')
    if select1 == 'Propotion of different income level':
        st.subheader("Propotion of different income level")
        st.markdown(
            "The majority of customers (approximately :green[49.1%]) fall within the lowest income bin, which represents a salary range of :green[-19475.726 to 4000268.0]. The proportion of customers in each income bin is as follows")
        st.dataframe(propIncome())

# 2nd part of the dashboard
st.sidebar.title("Explore and Visualize")
select2 = st.sidebar.selectbox('Select a type of feature or the target',
                               ['Percentage of Churn', 'Percentage of Credit Card Customers', 'Churn By Country',
                                'Social features', 'Numerical features', 'Bivariate', 'Visualising correlation coefficient', 'Proportion of churn and remaining customer', 'Visualising outlier with Boxplot'], key='3')

if not st.sidebar.checkbox("Hide", True, key='4'):
    st.markdown("### 2. Descriptive statistics")
    if select2 == 'Percentage of Churn':
        fig = piecharts(subtype=0)
        st.markdown("Hence, around :orange[22%] of consumers have churned. Therefore, the baseline model may predict that :orange[22%] of consumers would leave. However, considering that :orange[22%] is a tiny amount, we must guarantee that the selected model successfully predicts this :orange[22%] since it is more critical for the bank to identify and retain this group than to anticipate the remaining clients reliably.")
    if select2 == 'Percentage of Credit Card Customers':
        fig = piecharts(subtype=3)
        st.markdown(
            "In this dataset, approximately :blue[81.6%] of the customers own a credit card, while remaining :orange[18.4%] does not own a credit card.")
    elif select2 == 'Social features':
        fig = piecharts(subtype=1)
    elif select2 == 'Financial features':
        fig = piecharts(subtype=2)
    elif select2 == 'Numerical features':
        varlist = histos()[1]
        select2a = st.selectbox('Select a numerical feature', varlist, key='5')
        fig = histos(select2a)[0]
    elif select2 == 'Bivariate':
        varlist1 = scatters()[1]
        select2b = st.selectbox('Select a feature', varlist1, key='6')
        select2c = st.selectbox('Select another feature', varlist1, key='7')
        fig = scatters(select2b, select2c)[0]
    elif select2 == 'Visualising correlation coefficient':
        fig = heatmap()
    elif select2 == 'Churn By Country':
        fig = churnByCountry()
    elif select2 == 'Proportion of churn and remaining customer':
        fig = propChurnAndRemain()
    elif select2 == 'Visualising outlier with Boxplot':
        fig = outlierWithBoxplot()
    st.pyplot(fig)

    if select2 == 'Visualising outlier with Boxplot':
        st.markdown(":blue[Age]: The median age for both groups is around 40, but the group that does not churn has a wider age distribution, with a lower quartile around 30 and an upper quartile close to 50. The lower quartile for the churn group is roughly 35, and the top quartile is about 51. There are anomalies in both groups.")
        st.markdown(":blue[Credit score]: The median credit score for both groups is roughly 650, but the group that doesn't churn has a larger range of credit scores, with a lower quartile credit score of around 620 and an upper quartile credit score of about 680. The lower quartile for the churn group is around 620, and the upper quartile is around 660. There are anomalies in both groups.The churn group has a median that is not clear and an upper quartile around 1.5, whereas the not churn group tends to have more goods with a median around 1.5 and an upper quartile close to 2. The non churn group does not contain any outliers, however the churn group does contain a small outlier.")
        st.markdown(":blue[Balance]: The groups that do not churn often have larger balances, with a median around 0.75 and an upper quartile close to 1, as opposed to the groups that do churn, which have a median that is obscure and an upper quartile close to 1.2. There are some outliers in the churn group but none in the group that does not churn.")
        st.markdown(":blue[Tenure]: The non churn group tends to have tenures that are longer, with a median near 5 and an upper quartile close to 6, whereas the churn group has a median that is not clear and an upper quartile close to 5. There are some outliers in both groups, but there are fewer outliers in the non-churn group. Salary ranges for both groups are roughly similar, with a median wage of about 2.3–2.5 million, an upper quartile of about 11–12 million, and a lower quartile of about 10 million. There are no outliers in any category.")
        st.markdown(":blue[Investment]: The values of both groups' investments varied widely, with no clear median and an upper quartile close to 1500. In both categories, there are a few anomalies. Annual taxes: Both groups have a wide range of annual taxes, none of which have a clear median. The upper quartile is close to 1500, while the lower quartiles are located around 0. In both categories, there are a few anomalies.")

#  2nd part : classification for churn prediction
# st.sidebar.title("Churn prediction")
# select2 = st.sidebar.slider(
#     'Select number of trees to build in the random forest model', 100, 10, key='6')

# if not st.sidebar.checkbox("Hide", True, key='7'):
#     st.markdown("### 2. Churn prediction")
#     st.markdown(
#         "Supervised classification evaluation using a Random Forest model")
#     if select2:
#         score, score_, report, confus_matrix, fig, fig2 = classify(
#             model=RandomForestClassifier(n_estimators=select2, max_features=None))
#         st.write('Score for train data:\n ', round(score, 2))
#         st.write('Score for test data:\n ', round(score_, 2))

#         st.write('Classification report:\n ')
#         st.text('>\n ' + report)
#         st.write('\n ')
#         st.write('\n ')
#         st.write('Confustion matrix:\n ')
#         st.table(confus_matrix)
#         st.write('\n ')
#         st.write('\n ')
#         st.pyplot(fig)
#         st.write('\n ')
#         st.write('\n ')
#         st.pyplot(fig2)


# #  3rd part : clustering strategy
# st.sidebar.title("Clustering for a marketing strategy")
# select3 = st.sidebar.slider(
#     'Select the upper bound of estimated probability to be attrited (in %):', 100, 20)
# select4 = st.sidebar.slider('Now select the lower bound:', 0, select3)

# if not st.sidebar.checkbox("Hide", True):
#     st.markdown("### 3. Clustering for a marketing strategy")
#     st.markdown("See how many customers are in the probability interval.")
#     if select3:
#         fig, a, b = probacluster(model=RandomForestClassifier(
#             n_estimators=50, max_features=None), up=select3, lo=select4)
#         st.write('Number of selected customers:\n ', a)
#         st.write('Proportion:\n ', b)
#         st.write('\n ')
#         st.pyplot(fig)
