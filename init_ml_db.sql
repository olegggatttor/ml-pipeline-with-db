DROP DATABASE IF EXISTS ml_pipe_db;
CREATE DATABASE ml_pipe_db;
USE ml_pipe_db;
CREATE TABLE Predicts (
    login TINYTEXT,
    features BLOB,
    target FLOAT,
    predict FLOAT
);