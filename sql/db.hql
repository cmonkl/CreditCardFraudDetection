DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;


SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

CREATE EXTERNAL TABLE transactions STORED AS AVRO LOCATION '/project/transactions' TBLPROPERTIES ('avro.schema.url'='/project/avsc/transactions.avsc');
CREATE EXTERNAL TABLE merchant STORED AS AVRO LOCATION '/project/merchant' TBLPROPERTIES ('avro.schema.url'='/project/avsc/merchant.avsc');
CREATE EXTERNAL TABLE cart_holder STORED AS AVRO LOCATION '/project/cart_holder' TBLPROPERTIES ('avro.schema.url'='/project/avsc/cart_holder.avsc');


-- Creating bucketing anf partitioning
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.enforce.bucketing=true;

-- Partitioning on category and bucketing on merchant_id
CREATE EXTERNAL TABLE merchants_part_buck(
    merchant varchar(50),
    merch_lat double,
    merch_long double,
    merchant_id int) 
    PARTITIONED BY (category varchar(50)) 
    CLUSTERED BY (merchant_id) into 5 buckets
    STORED AS AVRO LOCATION '/project/merchant_part_buck' 
    TBLPROPERTIES ('AVRO.COMPRESS'='SNAPPY');
    
INSERT INTO merchants_part_buck partition (category) SELECT * FROM merchant;


-- Partitioning on is_fraud
CREATE EXTERNAL TABLE transactions_part(
index int,
trans_date_trans_time timestamp,
amt decimal(10,2),
trans_num VARCHAR(32),
unix_time integer,,
merchant_id int,
cart_holder_id int
)
PARTITIONED BY (is_fraud int)
STORED AS AVRO LOCATION '/project/transactions_part' 
TBLPROPERTIES ('AVRO.COMPRESS'='SNAPPY');

INSERT INTO transactions_part partition (is_fraud) SELECT * FROM transactions;

