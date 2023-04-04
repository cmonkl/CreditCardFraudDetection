#!/bin/bash

psql -U postgres -f sql/db.sql

sed '2i\host all all 0.0.0.0/0 trust\n' /var/lib/pgsql/data/pg_hba.conf
systemctl restart postgresql

wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar --no-check-certificate
cp  postgresql-42.6.0.jar /usr/hdp/current/sqoop-client/lib/
 
sqoop import-all-tables \
    -Dmapreduce.job.user.classpath.first=true \
    --connect jdbc:postgresql://localhost/credit_fraud \
    --username postgres \
    --warehouse-dir /project \
    --as-avrodatafile \
    --compression-codec=snappy \
    --m 1

hdfs dfs -mkdir -p /project/avsc    
hdfs dfs -put *.avsc /project/avsc
