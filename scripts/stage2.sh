#!/bin/bash

hive -f sql/db.hql

hive -f sql/queries.hql

echo "category,count" > output/q1.csv
cat /root/query_results/q1/* >> output/q1.csv

echo "category,difference" > output/q2.csv
cat /root/query_results/q2/* >> output/q2.csv

echo "state,difference" > output/q3.csv
cat /root/query_results/q3/* >> output/q3.csv

echo "gender,gender_fraud,gender_not_fraud" > output/q4.csv
cat /root/query_results/q4/* >> output/q4.csv

echo "is_fraud,avg_amt" > output/q5.csv
cat /root/query_results/q5/* >> output/q5.csv

echo "hour,hour_fraud,hour_not_fraud" > output/q6.csv
cat /root/query_results/q6/* >> output/q6.csv

echo "month,month_fraud,month_not_fraud" > output/q7.csv
cat /root/query_results/q7/* >> output/q7.csv

echo "age,age_fraud,age_not_fraud" > output/q8.csv
cat /root/query_results/q8/* >> output/q8.csv



