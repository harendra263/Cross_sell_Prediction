import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":

    df = pd.read_csv("input/modified_df.csv")
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.Response.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train, valid) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid, "kfold"] = fold
    
    df.to_csv("input/train_folds.csv", index=False)
        