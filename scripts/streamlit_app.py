import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
# from shapely.geometry import  Point

transactions = pd.read_csv("data/transactions.csv")
cart_holder = pd.read_csv("data/cart_holder.csv")
merchant = pd.read_csv("data/merchant.csv")

# reading queries results
q1 = pd.read_csv("output/q1.csv")
q2 = pd.read_csv("output/q2.csv")
q3 = pd.read_csv("output/q3.csv")
q4 = pd.read_csv("output/q4.csv")
q5 = pd.read_csv("output/q5.csv")
q6 = pd.read_csv("output/q6.csv")
q7 = pd.read_csv("output/q7.csv")
q_amt = pd.read_csv("output/q_amt.csv")
q_job = pd.read_csv("output/q_job.csv")
q_recency_fraud = pd.read_csv("output/q_recency_fraud.csv")
q_recency_not_fraud = pd.read_csv("output/q_recency_not_fraud.csv")
q_feat_importance = pd.read_csv("output/feat_importance.csv")
q_final_mod_thresh = pd.read_csv("output/final_model_thresh.csv")

st.markdown('---')
st.title("Big Data Project \n Zakirova Ainura, Irina Maltseva")
st.title("Credit Card Fraud Detection")
st.image("https://avatars.dzeninfra.ru/get-zen_doc/3927246/pub_6236f6a1f0553f30e967bf7b_6236f8f2b4e657033d8b4031/scale_1200",
         caption="Creadit Card Fraud Detection", width=600)

url = "https://nilsonreport.com/upload/content_promo/NilsonReport_Issue1209.pdf"
st.write(
    "Losses related to credit card fraud will grow to \$43 billion within five years and climb to \$408.5 billion globally within the next decade, according to a recent [Nilson Report](%s)— meaning that credit card fraud detection has become more important than ever. That is why it is crucial for credit card companies to be able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase." % url)

st.header("Data Characteristics")
st.subheader(f"Table 1. - Transactions - {len(transactions)} rows")
# transactions_description = transactions.describe()
types = transactions.dtypes
types['trans_num'] = 'string'
types['trans_date_trans_time'] = 'int64'
d = {'feature type': types}
transactions_types = pd.DataFrame(data=d, index=transactions.columns)
st.table(transactions_types)

st.write("**Descriptions of features:**")
st.write("●index - Unique Identifier for each row;")
st.write("●trans_date_trans_time - Transaction DateTime in unix format;")
st.write("●amt - Amount of Transaction;")
st.write("●trans_num - Transaction Number;")
st.write("●is_fraud - Fraud Flag <--- Target Class;")
st.write("●merchant_id - unique identifier for merchant;")
st.write("●cart_holder_id -  unique identifier for card holder.")

st.subheader(f"Table 2. - Merchant - {len(merchant)} rows")
# transactions_description = transactions.describe()
types = merchant.dtypes
types['merchant'] = 'string'
types['category'] = 'string'
d = {'feature type': types}
merchant_types = pd.DataFrame(data=d, index=merchant.columns)
st.table(merchant_types)

st.write("**Descriptions of features:**")
st.write("●merchant - Merchant Name;")
st.write("●category - Category of Merchant;")
st.write("●merch_lat - Latitude Location of Merchant;")
st.write("●merch_long - Longitude Location of Merchant;")
st.write("●merchant_id - unique identifier for merchant.")

st.write(
    f"It is important to mention, that even if merchant table has {len(merchant)} rows, actually we have only 693 unique merchants. However, each row is represented also by latitude and longitude, which make each row unique.")
st.subheader(f"Table 3. - Card holder - {len(cart_holder)} rows")
# transactions_description = transactions.describe()
types = cart_holder.dtypes
cat_feat = ['first', 'last', 'gender', 'street', 'city', 'state', 'job']
for feat in cat_feat:
    types[feat] = 'string'
d = {'feature type': types}
merchant_types = pd.DataFrame(data=d, index=cart_holder.columns)
st.table(merchant_types)

st.write("**Descriptions of features:**")
st.write("●cc_num - Credit Card Number of Customer;")
st.write("●first - First Name of Credit Card Holder;")
st.write("●last - Last Name of Credit Card Holder;")
st.write("●gender - Gender of Credit Card Holder;")
st.write("●street - Street Address of Credit Card Holdergender - Gender of Credit Card Holder;")
st.write("●city - City of Credit Card Holder;")
st.write("●state - State of Credit Card Holder;")
st.write("●zip - Zip of Credit Card Holder;")
st.write("●lat - Latitude Location of Credit Card Holder;")
st.write("●long - Longitude Location of Credit Card Holder;")
st.write("●city_pop - Credit Card Holder's City Population;")
st.write("●job - Job of Credit Card Holder;")
st.write("●dob - Date of Birth of Credit Card Holder;")
st.write("●cart_holder_id -  unique identifier for card holder.")

st.header("Exploratory Data Analysis (EDA)")


st.subheader("Fig. 1 - Distribution of category for fraud=1")
q1_plot = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(q1['category'], q1['count'], color='maroon',
        width=0.4)
plt.xlabel('category')
plt.ylabel('count')
plt.xticks(rotation=90)
st.write(q1_plot)
st.write("**Insight:**")
st.write("Most frauds occured in categories of shopping_net and grocery_pos.")

st.subheader(
    "Fig. 2 - The Percentage Difference of Fraudulent over Non-Fraudulent Transations in Each Spending Category")
st.write("We examine in which spending categories fraud happens most predominantly. To do this, we first calculate the distribution in normal transactions and then the the distribution in fraudulent activities. The difference between the 2 distributions will demonstrate which category is most susceptible to fraud. For example, if 'grocery_pos' accounts for 50\% of the total in normal transactions and 50\% in fraudulent transactions, this doesn't mean that it is a major category for fraud, it simply means it is just a popular spending category in general. However, if the percentage is 10\% in normal but 30\% in fraudulent, then we know that there is a pattern.")
q2_plot = alt.Chart(q2).mark_bar().encode(
    y='category',
    x='difference',
    color=alt.condition(
        alt.datum.difference > 0,
        alt.value("darkred"),
        alt.value("lightgrey"))
).properties(
    width=400,
    height=400
)
st.write(q2_plot)
st.write("**Insight:**")
st.write("Some spending categories indeed see more fraud than others. Fraud tends to happen more often in 'Shopping_net', 'Grocery_pos', and 'misc_net' while 'home' and 'kids_pets' among others tend to see more normal transactions than fraudulent ones.")


st.subheader(
    "Fig. 3 - The Percentage of Fraudulent over Non-Fraudulent Transcations in Each State")
st.write("Here we will explore which geographies are more prone to fraud. We will use the same methodology as in Figure 2, where we will calculate the difference in geographical distribution between the 2 transaction types.")
q3_plot = alt.Chart(q3).mark_bar().encode(
    y='state',
    x='difference',
    color=alt.condition(
        alt.datum.difference > 0,
        alt.value("darkred"),
        alt.value("lightgrey"))
).properties(
    width=400,
    height=500
)
st.write(q3_plot)
st.write("**Insight:**")
st.write("As can be seen, NY and OH among others have a higher percentage of fraudulent transactions than normal ones, while TX and MT are the opposite. However, it should be pointed out that the percentage differences in those states are not very significant but a correlation does exist.")


st.subheader("Fig. 4 - Gender vs Fraud")
st.write("We will examine whether one gender is more susceptible to fraud than the other.")


fig, ax1 = plt.subplots(1, 2)
ax1[0].pie(q4['gender_fraud'], labels=q4['gender'], autopct='%1.1f%%',
           shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax1[0].axis('equal')
ax1[0].set_title("Fraud=1")


ax1[1].pie(q4['gender_not_fraud'], labels=q4['gender'], autopct='%1.1f%%',
           shadow=True, startangle=90)
ax1[1].axis('equal')
ax1[1].set_title("Fraud=0")

st.pyplot(fig)
st.write("**Insight:**")
st.write("In this case, we do not see a clear difference between both genders. Data seem to suggest that females and males are almost equally susceptible (50%) to transaction fraud. Gender is not very indicative of a fraudulent transaction.")


st.subheader("Fig. 5 - Hourly Trend")
st.write("How do fraudulent transactions distribute on the temporal spectrum? Is there an hourly, monthly, or seasonal trend? We can use the transaction time column to answer this question.")

width = 0.4
fig_q5, ax = plt.subplots()

# creating the bar plot
ax.bar(q5['hour'], q5['hour_fraud'], color='maroon',
       width=0.4)
ax.bar(q5['hour']+width, q5['hour_not_fraud'], width=0.4)
ax.legend(['Fraud', 'Not Fraud'])
ax.set_xlabel("Hour")
ax.set_ylabel("Percentage")
ax.set_xticks(range(0, 24))
st.pyplot(fig_q5)
st.write("**Insight:**")
st.write("While normal transactions distribute more or less equally throughout the day, fraudulent payments happen disproportionately around midnight when most people are asleep.")


st.subheader("Fig. 6 - Monthly Trend")

width = 0.3

fig_q6, ax = plt.subplots()

# creating the bar plot
ax.bar(q6['month'], q6['month_fraud'], color='maroon',
       width=0.4)
ax.bar(q6['month'] + width, q6['month_not_fraud'], width=0.4)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend(['Fraud', 'Not Fraud'])
ax.set_xlabel("Month")
ax.set_ylabel("Percentage")
st.pyplot(fig_q6)
st.write("**Insight:**")
st.write("While normal payments peak around December (Christmas), and then late spring to early summer, fraudulent transactions are more concentrated in Jan-May. There is a clear seasonal trend.")


st.subheader("Fig. 7 - Age vs Fraud")
st.write("Are older people more prone to credit card fraud? Or is it the other way around? Given the birthday info, we can calculate the age of each card owner accroding the transaction date and see whether a trend exists.")

temp = cart_holder.merge(transactions, on='cart_holder_id', how='inner')
temp['age'] = pd.to_datetime(temp['trans_date_trans_time'],
                             unit='s').dt.year-pd.to_datetime(temp['dob'], unit='s').dt.year

fig_q7, ax = plt.subplots()
ax = sns.kdeplot(x='age', data=temp, hue='is_fraud', common_norm=False)
ax.set_xlabel('Credit Card Holder Age')
ax.set_ylabel('Density')
plt.xticks(np.arange(0, 110, 5))
plt.title('Age Distribution in Fraudulent vs Non-Fraudulent Transactions')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

st.pyplot(fig_q7)
st.write("**Insight:**")
st.write("The age distribution is visibly different between 2 transaction types. In normal transactions, there are 2 peaks at the age of 37-38 and 49-50, while in fraudulent transactions, the age distribution is a little smoother and the second peak does include a wider age group from 50-65. This does suggest that older people are potentially more prone to fraud.")


st.subheader("Fig. 8 - Number of Credit Card Frauds by Job")

fig_q_job, ax = plt.subplots()
palette = sns.cubehelix_palette(n_colors=10, start=0, reverse=True, rot=-0.3)
ax = sns.barplot(data=q_job, y="job", x="num_trans", palette=palette)

st.pyplot(fig_q_job)
st.write("**Insight:**")
st.write("We see that people who are working on the job 'Materials engineer' faced with more frauds than people in other jobs.")


st.subheader("Fig. 9 - Transaction amount vs Fraud")
st.write("Here we will examine how the distrition of transaction amount differs between fraudulent and normal activities. As there are extreme outliers in transaction amount, and the 99 percentile is around \$546, we subset the data for any transaction amounts below \$1,000 to make the visualizations more readable.")


fig_q_amt, ax = plt.subplots()
ax = sns.histplot(x='amt', data=q_amt, hue='is_fraud',
                  stat='percent', multiple='dodge', common_norm=False, bins=25)
ax.set_ylabel('Percentage in Each Type')
ax.set_xlabel('Transaction Amount in USD')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

st.pyplot(fig_q_amt)
st.write("**Insight:**")
st.write("While normal transactions tend to be around \$200 or less, we see fraudulent transactions peak around \$300 and then at the \$800-\$1000 range.")


st.subheader("Fig. 10 - Recency vs Fraud")
st.write("It is very interesting to find out whether some pattern exists in terms of time between transactions that relates to fraudulent activity.")


fig, ax1 = plt.subplots(2, figsize=[13, 10])
sns.barplot(ax=ax1[0], data=q_recency_fraud, y=q_recency_fraud['recency_segment'],
            x=q_recency_fraud['num_transactions'], palette=palette)
ax1[0].set_title("Fraud=1")


ax1[1] = sns.barplot(data=q_recency_not_fraud, y=q_recency_not_fraud['recency_segment'],
                     x=q_recency_not_fraud['num_transactions'], palette=palette)
ax1[1].set_title("Fraud=0")

st.pyplot(fig)
st.write("**Insight:**")
st.write("We can notice that there is a clear pattern between frequency of transactions and fraud. Specifically, when fraud is present, the majority of transactions occur within a one-hour interval, whereas when fraud is absent, the majority of transactions occur within a six-hour interval.")


st.header("Predictive Data Analysis (PDA)")

st.subheader("Feature Selection")

st.write("After exploring the data, features should be selected. For selection of numerical features we used [UnivariateFeatureSelector](%s), which operates on categorical/continuous labels with categorical/continuous features. We set continuos featureType and specified 'fwe' score function, that chooses all features whose p-values are below a threshold. The threshold is scaled by 1/numFeatures, thus controlling the family-wise error rate of selection." % "https://spark.apache.org/docs/latest/ml-features.html#univariatefeatureselector")
st.write("After applying UnivariateFeatureSelector, we obtained the following numerical features:")
st.write("●trans_hour_sin;")
st.write("●trans_hour_cos;")
st.write("●trans_month_sin;")
st.write("●trans_month_cos;")
st.write("●trans_day_sin;")
st.write("●trans_day_cos;")
st.write("●amt;")
st.write("●age;")
st.write("●recency.")
st.write("Trans_hour_sin, trans_hour_cos, trans_month_sin, trans_month_cos, trans_day_sin, trans_day_cos are new features, which are transformations from initial hour, month, and day features acccordingly.")

st.write("Regarding categorical features we chose **'category'** for the further precitions, because according to Figure 1, there is a clear dependence between some categories and 'is_fraud'. At the same time other categorical features have a lot of unique categories, i.e. high cardinality. That is why we decided to drop these features for now. Moreover, such features as 'first' and 'last', which is a name and surname for card holder, can lead to issue of privacy.")


st.subheader("Models")
st.write("It is important to check the balance between values in target. As we have binary classification, we have only 2 labels: 0 and 1. The distribution of labels is the following:")
count = transactions.groupby('is_fraud').count()['index'].values
st.write(f"●Fraud=1 :{count[1]}")
st.write(f"●Fraud=0 : {count[0]}")
st.write("We see that data is imbalanced. Therefore, we will consider this finding when we will choose the metrics. For example, we will not use Accuracy metric, because it does not take into account the inbalance of the dataset.")
st.write("As we are dealing with classification problem, we utilized three classification models, such as Logistic Regression, Decision Tree, and Gradient Boosting. Each model was fine-tuned using GridSearch, and the best parameters for the model were chosen.")

st.write("We will show the comparison of the models after fine-tuning, i.e. on their best parameters:")
st.subheader("Table 4. - Comparison of models.")
perform = {'Precision': [0.071, 0.79, 0.917], 'Recall': [0.004, 0.49, 0.662],
           'F1-Score': [0.008, 0.605, 0.769], 'Area under PR curve': [0.0386, 0.59, 0.763]}
performance = pd.DataFrame(data=perform, index=[
                           "Logistic Regression", "Decision Tree", "Gradient Boosting"])
st.table(performance)

st.write("We can notice that Gradient Boosting model showed the highest results in comparison with other models.")

st.subheader("Feature Importance")
st.write("We extracted feature importance from the best model - Gardient Boosting. The results of first 20 features are shown below:")

st.subheader("Fig. 11 - Feature Importance")
fig, ax1 = plt.subplots()
palette_2 = sns.cubehelix_palette(n_colors=20, start=0, reverse=True, rot=-0.3)
ax1 = sns.barplot(data=q_feat_importance,
                  y=q_feat_importance['score'], x=q_feat_importance['name'], palette=palette_2)
plt.xticks(rotation=90)
st.pyplot(fig)
st.write("We can observe that 'amt', 'age','trans_hour_cos', 'recency', and 'trans_hour_sin' have the highest score in comparison with other features. In terms of transactions it really makes sense. Given feature importance from the model we can understand how decisions were made and why we got particular results.")


st.header("Final results")
st.write("We compared the performance of three models: Logistic Regression, Decision Tree, and Gradient Boosting. We found out that Gradient Boosting performs the best. Using fine-tuning of parameters with GridSearch, we tried different combinations of parameters maxDepth, stepSize, maxIter and obtained such combination, which gives the highest score. The final model parameters are shown below:")
parameters = {'Parameters': ['maxDepth', "stepSize",
                             'maxIter'], 'Values': ['7', '0.1', '20'], 'Possible values': ['[3, 7]', '[0.06, 0.1]', '[10, 20]']}
st.subheader("Table 5. - Gradient Boosting parameters.")
params = pd.DataFrame(data=parameters)
st.table(params)

st.write("Let us look at the results in terms of confusion matrix:")
st.subheader("Figure 12. - Confusion matrix")
length = int(0.3*len(transactions))
conf_matrix = np.array([[386868, 785], [139, 1537]])/length
fig, ax = plt.subplots()
ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

st.pyplot(fig)

st.write("We see that True Negative (TN) rate is very high. This is because of the imbalance of our data. Even if TN is high, we should consider errors that our model makes since the the goal of this project is to predict possible frauds in order to reduce financial losses for credit card companies. For example:")
st.write("●The cost of fraud (FN) - predicting not fraud when the transaction was fraud.\nConsequences: fraud is reported, card is locked, card company liable for refunding.")
st.write("●the cost of human labour for investigating the flagged transactions (FP)  - predicting fraud when the transaction was not fraud.\nConsequences: card is locked, card owner calls and explains the mistake, then there is short investigation, card is unlocked.")
st.write("Based on the consequences of false negatives, we can conclude that our objective should be to minimize them as much as possible. However, we should avoid going too far and reducing the number of transactions being blocked to an unreasonable extent.")
st.write("Thus, the loss in terms of money can be calculated in the following way:")
st.latex("Cost(i) = cost(FP) * count(FP) + cost(FN) * count(FN),")
st.write("where cost(FP)=\$4 according to our assumptions (more detailed in report), and cost(FN) is the amount of transaction.")
st.write("We trained our model in different threshold values to obtain such value, which minimizes loss described above. The following Figure represents results:")
st.subheader("Figure 13. - Threshold of the model")

fig, ax = plt.subplots()
ax = plt.plot(q_final_mod_thresh['threshold'], q_final_mod_thresh['loss'])
plt.xlabel('threshold')
plt.ylabel('Loss')
st.pyplot(fig)
st.write("According to given results, we obtained threshold = 0.05 as one which minimizes the costs the most. With this threshold we measured the losses on test data using the formula above and profit by subtracting losses from the total amount of transactions:")
st.latex("Losses = 58903.58")
st.latex("Total \quad amt = 1247884.17 ")
st.latex("Profit = Total \quad amt - Losses = 1138284.95")
st.write("We can see that losses account to only 4\% of the total transaction value, which is quite satisfactory. It is very important to measure results in terms of money, because our project is business oriented.")

st.write("To summarize, we built a model, which is focusing on predicting frauds in credit cards. Our goal was to minimize the risks of credit card fraud. By incorporating credit card fraud detection system, our customers can reduce the risk of processing fraud transactions and minimize efforts for credit card detection monitoring.")
