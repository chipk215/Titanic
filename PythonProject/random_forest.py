import common
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def process_training_data(df_train):
    missing_features = common.get_missing_feature_list(df_train)
    df_train = common.handle_missing_features(missing_features, df_train)
    # replace Sex attribute with numeric encoding
    df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked'], drop_first=True)
    y_predict = df_train['Survived']

    # Drop PassengerId, Survived, Name and Ticket features for use in model
    predict_features = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

    train_X, val_X, train_y, val_y = train_test_split(predict_features, y_predict,  random_state=0)
    tree_count = 1000
    rf_classifier = RandomForestClassifier(tree_count, n_jobs=-1, max_features='sqrt',
                                           min_samples_leaf=1, bootstrap=True, random_state=42)
    rf_classifier.fit(train_X, train_y)
    score = rf_classifier.score(train_X, train_y)
    print("Validation Model Score= ", score)
    survivor_predictions = rf_classifier.predict(val_X)
    print(confusion_matrix(val_y, survivor_predictions))
    tn, fp, fn, tp = confusion_matrix(val_y, survivor_predictions).ravel()
    training_accuracy = common.compute_accuracy(tn, tp, val_X.shape[0])
    print("Validation Training Accuracy =", training_accuracy)

    training_records = df_train.loc[val_X.index.values]
    passengerIds = training_records['PassengerId']

    validation_result = pd.DataFrame()
    validation_result['PassengerId'] = passengerIds
    validation_result['Survived'] = val_y
    validation_result['Prediction'] = survivor_predictions

    # train model with all of the training samples
    rf_classifier.fit(predict_features, y_predict)
    score = rf_classifier.score(predict_features,y_predict)
    print("Training Model Score= ", score)
    return rf_classifier, predict_features, validation_result


def process_test_data(df_test, rf_classifier, training_features):
    # Predict survivors using test data
    missing_features = common.get_missing_feature_list(df_test)

    df_test = common.handle_missing_features(missing_features, df_test)
    # replace categorical features with numeric encoding
    df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
    # Drop PassengerId, Name and Ticket features for use in model
    predict_features = df_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    # Ensure training features and test features match.
    # The issue is that test data may not have a specific instance for a categorical feature that
    # has been represented as a dummy
    test_features = list(predict_features)
    missing_features = list(set(training_features).difference(test_features))
    for feature in missing_features:
        predict_features[feature] = 0

    test_predictions = rf_classifier.predict(predict_features)

    passenger_ids = df_test['PassengerId']
    print(type(test_predictions), test_predictions.shape)
    result = pd.DataFrame(passenger_ids)
    result['Survived'] = test_predictions

    result.to_csv('results/test_results.csv', encoding='utf-8', index=False)


def run_training_model():
    df_train, df_test = common.read_data_files()
    rf_classifier, rf_training_features, rf_results = process_training_data(df_train)

    importances = rf_classifier.feature_importances_
    feature_importances = common.join_feature_name_with_importance_value(rf_training_features, importances)

    return rf_classifier, rf_training_features, rf_results


def run_main():
    rf_classifier, rf_training_features, rf_results = run_training_model()


    df = pd.DataFrame()
    df['PassengerId'] = rf_results['PassengerId']
    df['Survived'] = rf_results['Survived']
    df['RF_Predict'] = rf_results['Prediction']
    df.to_csv('results/rf_validation_results.csv', encoding='utf-8', index=False)

    # re-read the data
    df_train, df_test = common.read_data_files()

    errors = df.query('Survived != RF_Predict')
    false_positives = errors.loc[errors['RF_Predict'] == 1]['PassengerId']
    list_false_positive_ids = false_positives.tolist()
    false_positive_passengers = df_train[df_train['PassengerId'].isin(list_false_positive_ids)]

    false_negatives = errors.loc[errors['RF_Predict'] == 0]['PassengerId']
    list_false_negative_ids = false_negatives.tolist()
    false_negative_passengers = df_train[df_train['PassengerId'].isin(list_false_negative_ids)]

    correct_predictions = df.query('Survived == RF_Predict')
    correct_passenger_ids = correct_predictions['PassengerId']
    list_correct_predictions = correct_passenger_ids.tolist()
    correct_passengers = df_train[df_train['PassengerId'].isin(list_correct_predictions)]

    # Predict survivors using test data
    # process_test_data(df_test, rf_classifier, training_features_list)
    pass


if __name__ == '__main__':
    # run_main()
    run_training_model()

