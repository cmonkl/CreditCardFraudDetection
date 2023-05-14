import pandas as pd
from datetime import datetime


transactions = pd.read_csv("data/transactions.csv", header=0)
transactions['trans_date_trans_time'] = transactions['trans_date_trans_time'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
transactions['trans_date_trans_time'] = (
    transactions['trans_date_trans_time'] - pd.Timestamp("1970-01-01"))/pd.Timedelta('1s')
transactions = transactions.drop('unix_time', axis=1)
transactions.trans_date_trans_time = transactions.trans_date_trans_time.astype(
    int)
transactions.to_csv("data/transactions.csv", index=False)

cart_holder = pd.read_csv("data/cart_holder.csv", header=0)
cart_holder['dob'] = cart_holder['dob'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d'))
cart_holder['dob'] = (cart_holder['dob'] -
                      pd.Timestamp("1970-01-01"))/pd.Timedelta('1s')
cart_holder.dob = cart_holder.dob.astype(int)
cart_holder.to_csv("data/cart_holder.csv", index=False)
