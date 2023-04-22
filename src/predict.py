import argparse
import pickle
import json
import numpy as np
import logging

from db import get_connection, INSERT_PREDICTIONS
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(prog='BikeSharingDemandRegression')
    parser.add_argument("--data", default="tests/func_samples.csv")
    parser.add_argument("--from_pretrained", default="data/r_forest.pickle")
    args = parser.parse_args()

    with open(args.from_pretrained, 'rb') as f:
        trainer = Trainer.from_pretrained(pickle.load(f), args.data, args.data)
        test_predictions = trainer.predict(trainer.get_train())

        assert np.allclose(test_predictions, trainer.get_train()['count'], rtol=0, atol=35), \
            (test_predictions, trainer.get_train()['count'])

    db = get_connection('root', 'pass', 'mldb', 3310, 'ml_pipe_db')
    try:
        cursor = db.cursor()

        cursor.execute("DESCRIBE Predicts")

        logging.info(cursor.fetchall())

        to_store = trainer.get_train()
        to_store_data = to_store.drop('count', axis=1).to_numpy()
        to_store_target = to_store['count']

        to_insert = [
            ('root', features.tobytes(), float(target), float(pred))
            for features, target, pred in zip(to_store_data, to_store_target, test_predictions)
        ]

        cursor.executemany(
            INSERT_PREDICTIONS,
            to_insert
        )
        db.commit()
        logging.info(f"{cursor.rowcount} rows was inserted.")
    finally:
        db.close()


if __name__ == '__main__':
    main()
