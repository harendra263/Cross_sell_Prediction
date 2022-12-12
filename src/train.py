import pandas as pd
import numpy as np
import os
from sklearn import metrics
import joblib

from . import dispatcher


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ["MODEL"]


FOLD_MAPPING = {
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3, 4, 5],
    2: [0, 1, 3, 4, 5],
    3: [0, 1, 2, 4, 5],
    4: [0, 1, 2, 3, 5]
}



if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    test = pd.read_csv(TEST_DATA)



    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.Response.values
    yvalid = valid_df.Response.values

    train_df = train_df.drop(["id", "kfold", "Response"], axis=1)
    valid_df = valid_df.drop(["id", "kfold", "Response"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    print(clf)
    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(clf, f'models/{MODEL}_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'models/{MODEL}_{FOLD}_columns.pkl')

