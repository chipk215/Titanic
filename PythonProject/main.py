import common
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def process_training_data(df_train):
    missing_features = common.get_missing_feature_list(df_train)
    df_train = common.handle_missing_features(missing_features, df_train)
    # replace Sex attribute with numeric encoding
    df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
    y_predict = df_train['Survived']

    # Drop PassengerId, Survived, Name and Ticket features for use in model
    predict_features = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)

    train_X, val_X, train_y, val_y = train_test_split(predict_features, y_predict,  random_state=214)
    tree_count = 1000
    rf_classifier = RandomForestClassifier(tree_count, n_jobs=-1, max_features='sqrt',
                                           min_samples_leaf=1, bootstrap=True, random_state=42)
    rf_classifier.fit(train_X, train_y)
    score = rf_classifier.score(train_X, train_y)
    print("Model score= ", score)
    survivor_predictions = rf_classifier.predict(val_X)
    print(confusion_matrix(val_y, survivor_predictions))
    tn, fp, fn, tp = confusion_matrix(val_y, survivor_predictions).ravel()
    training_accuracy = common.compute_accuracy(tn, tp, val_X.shape[0])
    print("Training accuracy =", training_accuracy)

    return rf_classifier, list(predict_features)


def process_test_data(df_test, rf_classifier, training_features):
    # Predict survivors using test data
    missing_features = common.get_missing_feature_list(df_test)

    # Drop PassengerId, Name and Ticket features for use in model

    df_test = common.handle_missing_features(missing_features, df_test)
    # replace Sex attribute with numeric encoding
    df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
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


def run_main():
    df_train, df_test = common.read_data_files()
    rf_classifier, training_features = process_training_data(df_train)
    # Predict survivors using test data
    # process_test_data(df_test, rf_classifier, training_features)


if __name__ == "__main__":
    run_main()
