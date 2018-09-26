import common
import random_forest


def run_main():
    df_train, df_test = common.read_data_files()
    rf_classifier, training_features = random_forest.process_training_data(df_train)
    # Predict survivors using test data
    # random_forest.process_test_data(df_test, rf_classifier, training_features)


if __name__ == "__main__":
    run_main()
