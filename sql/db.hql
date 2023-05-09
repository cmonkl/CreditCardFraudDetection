DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;


SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

CREATE EXTERNAL TABLE transactions STORED AS AVRO LOCATION '/project/transactions' TBLPROPERTIES ('avro.schema.url'='/project/avsc/transactions.avsc');
CREATE EXTERNAL TABLE merchant STORED AS AVRO LOCATION '/project/merchant' TBLPROPERTIES ('avro.schema.url'='/project/avsc/merchant.avsc');
CREATE EXTERNAL TABLE cart_holder STORED AS AVRO LOCATION '/project/cart_holder' TBLPROPERTIES ('avro.schema.url'='/project/avsc/cart_holder.avsc');


-- Creating partitioning
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

-- Partitioning on is_fraud
CREATE EXTERNAL TABLE transactions_part(
index int,
trans_date_trans_time int,
amt decimal(10,2),
trans_num VARCHAR(32),
merchant_id int,
cart_holder_id int
)
PARTITIONED BY (is_fraud int)
STORED AS AVRO LOCATION '/project/transactions_part' 
TBLPROPERTIES ('AVRO.COMPRESS'='SNAPPY');


INSERT INTO transactions_part partition (is_fraud=0) SELECT index, trans_date_trans_time, amt, trans_num, merchant_id, cart_holder_id FROM transactions WHERE is_fraud=0;
INSERT INTO transactions_part partition (is_fraud=1) SELECT index, trans_date_trans_time, amt, trans_num, merchant_id, cart_holder_id FROM transactions WHERE is_fraud=1;


