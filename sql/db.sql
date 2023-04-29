DROP DATABASE IF EXISTS credit_fraud;
CREATE DATABASE credit_fraud;

\c credit_fraud;

START TRANSACTION;

DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    index integer NOT NULL PRIMARY KEY,
    trans_date_trans_time timestamp,
    amt decimal(10, 2),
    trans_num VARCHAR(32),
    unix_time integer,
    is_fraud integer NOT NULL,
    merchant_id integer NOT NULL,
    cart_holder_id integer NOT NULL
);

DROP TABLE IF EXISTS merchant;

CREATE TABLE merchant (
    merchant VARCHAR (100),
    category VARCHAR (50),
    merch_lat double precision,
    merch_long double precision,
    merchant_id integer NOT NULL PRIMARY KEY
);   
    
DROP TABLE IF EXISTS cart_holder;

CREATE TABLE cart_holder(
    cc_num VARCHAR (20),
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
    cart_holder_id integer NOT NULL PRIMARY KEY
);

ALTER TABLE transactions ADD CONSTRAINT fk_trans_merch FOREIGN KEY(merchant_id) REFERENCES merchant (merchant_id);

ALTER TABLE transactions ADD CONSTRAINT fk_trans_cart_hold FOREIGN KEY(cart_holder_id) REFERENCES cart_holder (cart_holder_id);

-- SET datestyle TO iso, ymd;

\COPY transactions FROM 'data/transactions.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

\COPY merchant FROM 'data/merchant.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

\COPY cart_holder FROM 'data/cart_holder.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

COMMIT;

SELECT * from transactions limit 5;
SELECT * from merchant limit 5;
SELECT * from cart_holder limit 5;

