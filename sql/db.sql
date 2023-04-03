DROP DATABASE IF EXISTS credit_fraud;
CREATE DATABASE credit_fraud;

\c credit_fraud;

START TRANSACTION;

DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    index integer NOT NULL PRIMARY KEY,
    trans_date_trans_time timestamp,
    cc_num VARCHAR (20),
    merchant VARCHAR (100),
    category VARCHAR (50),
    amt decimal(10, 2),
    first VARCHAR (20),
    last VARCHAR (30),
    gender VARCHAR (1),
    street VARCHAR (100),
    city VARCHAR (50),
    state VARCHAR (2),
    zip integer,
    lat double precision,
    long double precision,
    city_pop integer,
    job VARCHAR(100),
    dob date,
    trans_num VARCHAR(32),
    unix_time integer,
    merch_lat double precision,
    merch_long double precision,
    is_fraud integer NOT NULL
);

-- SET datestyle TO iso, ymd;

\COPY transactions FROM 'data/fraudTrain.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

COMMIT;

SELECT * from transactions limit 5;
