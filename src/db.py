import mysql.connector
import logging


def get_connection(user, password, host, port, database):
    return mysql.connector.connect(user=user, password=password,
                                   host=host, database=database, port=port)


HOST_NAME = "mldb"
DB_NAME = "ml_pipe_db"
INSERT_PREDICTIONS = "INSERT INTO Predicts (login, features, target, predict) VALUES (%s, %s, %s, %s)"
SELECT_ALL_PREDICTS = "SELECT * FROM Predicts"
