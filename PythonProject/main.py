import common
import pandas as pd
import gradient_boosting
import random_forest
import ada_boost


def run_main():
    df_train, df_test = common.read_data_files()

    gb_classifier, gb_training_features, gradient_results = gradient_boosting.process_training_data(df_train)

    df_train, df_test = common.read_data_files()
    rf_classifier, rf_training_features, rf_results = random_forest.process_training_data(df_train)

    df_train, df_test = common.read_data_files()
    ada_classifier, ada_training_features, ada_results = ada_boost.process_training_data(df_train)

    df_combined = pd.DataFrame()
    df_combined['PassengerId'] = gradient_results['PassengerId']
    df_combined['Survived'] = gradient_results['Survived']
    df_combined['Gradient_Predict'] = gradient_results['Prediction']
    df_combined['Ada_Predict'] = ada_results['Prediction']
    df_combined['RF_Predict'] = rf_results['Prediction']
    df_combined['Sum'] = df_combined['Ada_Predict'] + df_combined['Gradient_Predict'] + df_combined['RF_Predict']
    df_combined['Vote'] = 0
    df_combined['Vote'][df_combined['Sum'] > 1] = 1

    df_combined.to_csv('results/combined_results.csv', encoding='utf-8', index=False)
    # Predict survivors using test data
    # random_forest.process_test_data(df_test, rf_classifier, training_features)


if __name__ == "__main__":
    run_main()
