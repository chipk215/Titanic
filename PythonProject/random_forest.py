import common
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def analyze_training_results(df_train, val_X, val_y,  survivor_predictions):
    training_records = df_train.loc[val_X.index.values]
    passengerIds = training_records['PassengerId']

    validation_result = pd.DataFrame()
    validation_result['PassengerId'] = passengerIds
    validation_result['Survived'] = val_y
    validation_result['Prediction'] = survivor_predictions

    df = pd.DataFrame()
    df['PassengerId'] = validation_result['PassengerId']
    df['Survived'] = validation_result['Survived']
    df['RF_Predict'] = validation_result['Prediction']
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


def run_model():
    df_train, df_test = common.read_data_files()
    missing_training_features = common.get_missing_feature_list(df_train)
    df_train = common.handle_missing_features(missing_training_features, df_train)
    y_predict = df_train['Survived']

    missing_test_features = common.get_missing_feature_list(df_test)
    df_test = common.handle_missing_features(missing_test_features, df_test)

    # Convert titles to categories
    df_train['Title'] = common.convert_titles_to_categories(df_train)
    df_train = pd.get_dummies(df_train, columns=['Title', 'Sex'], drop_first=True)
    df_test['Title'] = common.convert_titles_to_categories(df_test)
    df_test = pd.get_dummies(df_test, columns=['Title', 'Sex'], drop_first=True)

    # Generate grouped tickets feature
    df_train = common.identify_group_tickets(df_train)
    df_test = common.identify_group_tickets(df_test)

    # Drop PassengerId, Name and Ticket features for use in model
    predict_features_train = df_train.drop(['PassengerId', 'Survived',
                                            'Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)
    predict_features_test = df_test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)

    # Fit the model and make training predictions
    train_X, val_X, train_y, val_y = train_test_split(predict_features_train, y_predict, random_state=42)
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

    common.display_important_features(rf_classifier, predict_features_train)

    # Predict test features with model
    # Ensure training features and test features match.
    # The issue is that test data may not have a specific instance for a categorical feature that
    # has been represented as a dummy
    test_features = list(predict_features_test)
    train_features = list(predict_features_train)
    missing_features = list(set(train_features).difference(test_features))
    for feature in missing_features:
        predict_features_test[feature] = 0

    test_predictions = rf_classifier.predict(predict_features_test)
    common.create_results_submission_file(df_test, test_predictions)

    return rf_classifier


if __name__ == '__main__':
    run_model()


