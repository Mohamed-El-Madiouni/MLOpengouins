import pengouins.data as data
import pengouins.model as model
import pengouins.registry as registry


def main():
    # Load data
    df = data.load_data(path="pengouins")

    # Split into features and target
    X, y = data.get_X_y(df, target_column="species")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = data.split_data(X, y)

    # Preprocess data
    X_train_processed = data.preprocess_data(X_train, fit=True)
    X_test_processed = data.preprocess_data(X_test, fit=False)

    logi = model.train_model(X_train_processed, y_train)

    score = model.evaluate_model(logi, X_test_processed, y_test)

    registry.save_model(logi, "logistic_regression_model.pkl")

    logi_loaded = registry.load_model("logistic_regression_model.pkl")


if __name__ == "__main__":
    main()
