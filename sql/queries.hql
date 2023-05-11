
USE projectdb;


-- Grouping on merchant category for is_fraud
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q1'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT m.category, COUNT(m.category)
FROM merchant_buck as m inner join transactions_part as t on m.merchant_id=t.merchant_id
WHERE t.is_fraud = 1
GROUP BY m.category;


-- Percentage distance between fraud and not fraud for category
WITH merchant_transactions as (SELECT m.category, t.is_fraud FROM merchant_buck as m inner join transactions_part as t on m.merchant_id=t.merchant_id),
cat_fraud as (SELECT category, COUNT(category) as count FROM merchant_transactions WHERE is_fraud = 1 GROUP BY category),
cat_not_fraud as (SELECT category, COUNT(category) as count FROM merchant_transactions WHERE is_fraud = 0 GROUP BY category),
sum_fraud as (SELECT SUM(cf.count) as sum FROM cat_fraud as cf),
sum_not_fraud as (SELECT SUM(cnf.count) as sum FROM cat_not_fraud as cnf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q2'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT cat_fraud.category, (cat_fraud.count/sf.sum - cat_not_fraud.count/snf.sum)*100 as difference FROM cat_fraud inner join cat_not_fraud on cat_fraud.category=cat_not_fraud.category, sum_fraud as sf, sum_not_fraud as snf;


-- Percentage distance between fraud and not fraud for state
WITH cart_holder_transactions as (SELECT ch.state, t.is_fraud FROM cart_holder_buck as ch inner join transactions_part as t on ch.cart_holder_id=t.cart_holder_id),
state_fraud as (SELECT state, COUNT(state) as count FROM cart_holder_transactions WHERE is_fraud = 1 GROUP BY state),
state_not_fraud as (SELECT state, COUNT(state) as count FROM cart_holder_transactions WHERE is_fraud = 0 GROUP BY state),
sum_fraud as (SELECT SUM(sf.count) as sum FROM state_fraud as sf),
sum_not_fraud as (SELECT SUM(snf.count) as sum FROM state_not_fraud as snf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q3'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT state_fraud.state, (state_fraud.count/sf.sum - state_not_fraud.count/snf.sum)*100 as difference FROM state_fraud inner join state_not_fraud on state_fraud.state=state_not_fraud.state, sum_fraud as sf, sum_not_fraud as snf;


-- Gender vs Fraud
WITH cart_holder_transactions as (SELECT ch.gender, t.is_fraud FROM cart_holder_buck as ch inner join transactions_part as t on ch.cart_holder_id=t.cart_holder_id), 
gender_fraud as (SELECT gender, COUNT(gender) as count FROM cart_holder_transactions WHERE is_fraud = 1 GROUP BY gender),
gender_not_fraud as (SELECT gender, COUNT(gender) as count FROM cart_holder_transactions WHERE is_fraud = 0 GROUP BY gender),
sum_fraud as (SELECT SUM(gf.count) as sum FROM gender_fraud as gf),
sum_not_fraud as (SELECT SUM(gnf.count) as sum FROM gender_not_fraud as gnf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q4' 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT gender_fraud.gender, gender_fraud.count/sf.sum*100, gender_not_fraud.count/snf.sum*100 FROM gender_fraud inner join gender_not_fraud on gender_fraud.gender=gender_not_fraud.gender, sum_fraud as sf, sum_not_fraud as snf;


-- Transactions amount vs Fraud
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q5' 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT is_fraud, AVG(amt)
FROM transactions_part
GROUP BY is_fraud;


-- Hourly trend
WITH transactions_time as (SELECT FROM_UNIXTIME(t.trans_date_trans_time) as time, t.is_fraud FROM transactions_part as t),
transactions_hours as (SELECT HOUR(t.time) as hour, t.is_fraud FROM transactions_time as t),
hour_fraud as (SELECT hour, COUNT(hour) as count FROM transactions_hours WHERE is_fraud=1 GROUP BY hour),
hour_not_fraud as (SELECT hour, COUNT(hour) as count FROM transactions_hours WHERE is_fraud=0 GROUP BY hour),
sum_fraud as (SELECT SUM(hf.count) as sum FROM hour_fraud as hf),
sum_not_fraud as (SELECT SUM(hnf.count) as sum FROM hour_not_fraud as hnf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q6' 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT hf.hour, hf.count/sf.sum*100, hnf.count/snf.sum*100
FROM hour_fraud as hf inner join hour_not_fraud as hnf on hf.hour=hnf.hour, sum_fraud as sf, sum_not_fraud as snf;


-- Monthly trend
WITH transactions_time as (SELECT FROM_UNIXTIME(t.trans_date_trans_time) as time, t.is_fraud FROM transactions_part as t),
transactions_month as (SELECT MONTH(t.time) as month, t.is_fraud FROM transactions_time as t),
month_fraud as (SELECT month, COUNT(month) as count FROM transactions_month WHERE is_fraud=1 GROUP BY month),
month_not_fraud as (SELECT month, COUNT(month) as count FROM transactions_month WHERE is_fraud=0 GROUP BY month),
sum_fraud as (SELECT SUM(mf.count) as sum FROM month_fraud as mf),
sum_not_fraud as (SELECT SUM(mnf.count) as sum FROM month_not_fraud as mnf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q7' 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT hf.month, hf.count/sf.sum*100, hnf.count/snf.sum*100
FROM month_fraud as hf inner join month_not_fraud as hnf on hf.month=hnf.month, sum_fraud as sf, sum_not_fraud as snf;


-- Age vs Fraud
WITH cart_holder_transactions as (SELECT t.trans_date_trans_time, t.is_fraud, ch.dob FROM cart_holder_buck as ch inner join transactions_part as t on ch.cart_holder_id=t.cart_holder_id),
cart_holder_age as (SELECT (YEAR(FROM_UNIXTIME(trans_date_trans_time))-YEAR(FROM_UNIXTIME(dob))) as age, is_fraud FROM cart_holder_transactions),
age_fraud as (SELECT age, COUNT(age) as count FROM cart_holder_age WHERE is_fraud = 1 GROUP BY age),
age_not_fraud as (SELECT age, COUNT(age) as count FROM cart_holder_age WHERE is_fraud = 0 GROUP BY age),
sum_fraud as (SELECT SUM(af.count) as sum FROM age_fraud as af),
sum_not_fraud as (SELECT SUM(anf.count) as sum FROM age_not_fraud as anf)
INSERT OVERWRITE LOCAL DIRECTORY '/root/query_results/q8'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT age_fraud.age, age_fraud.count/sf.sum, age_not_fraud.count/snf.sum FROM age_fraud inner join age_not_fraud on age_fraud.age=age_not_fraud.age, sum_fraud as sf, sum_not_fraud as snf;
