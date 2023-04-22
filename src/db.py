import mysql.connector
import logging


def get_connection(user, password, host, port, database):
    return mysql.connector.connect(user=user, password=password,
                                   host=host, database=database, port=port)


INSERT_PREDICTIONS = "INSERT INTO Predicts (login, features, target, predict) VALUES (%s, %s, %s, %s)"
