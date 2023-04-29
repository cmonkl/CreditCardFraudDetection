DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;


SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

CREATE EXTERNAL TABLE transactions STORED AS AVRO LOCATION '/project/transactions' TBLPROPERTIES ('avro.schema.url'='/project/avsc/transactions.avsc');
CREATE EXTERNAL TABLE merchants STORED AS AVRO LOCATION '/project/merchant' TBLPROPERTIES ('avro.schema.url'='/project/avsc/merchant.avsc');
CREATE EXTERNAL TABLE cart_holders STORED AS AVRO LOCATION '/project/cart_holder' TBLPROPERTIES ('avro.schema.url'='/project/avsc/cart_holder.avsc');


-- For checking the content of tables
SELECT * FROM transactions limit 5;
SELECT * FROM merchants limit 5;
SELECT * FROM cart_holders limit 5;
