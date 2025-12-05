import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from model_utils import TRAIN_URL, TEST_URL, make_pipeline


def load_data():
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)
    return df_train, df_test


def main():
    df_train, df_test = load_data()

    y_train = df_train["selling_price"]
    y_test = df_test["selling_price"]

    # Базовый пайплайн: скейлим только числовые
    pipeline = make_pipeline(scale_cat=False)
    pipeline.fit(df_train, y_train)

    train_pred = pipeline.predict(df_train)
    test_pred = pipeline.predict(df_test)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = artifacts_dir / "model_bundle.pkl"
    joblib.dump(
        {
            "pipeline": pipeline,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mse_train": mse_train,
            "mse_test": mse_test,
        },
        bundle_path,
    )

    print(f"Saved model to {bundle_path}")
    print("Baseline (num scaled):")
    print(f"  R2 train/test: {r2_train:.3f} / {r2_test:.3f}")
    print(f"  MSE train/test: {mse_train:,.1f} / {mse_test:,.1f}")


if __name__ == "__main__":
    main()
