import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def read_data_files():
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    return df_train, df_test


def get_missing_feature_list(df):
    observation_count = df.shape[0]
    feature_counts = df.count()
    missing_feature_list = []
    for idx, value in feature_counts.iteritems():
        if value != observation_count:
            missing_feature_list.append(idx)

    return missing_feature_list


def handle_embark_missing_values(df):
    """
    Majority of passengers embarked in Southampton
    :param df:
    :return:
    """
    df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'
    return df


def handle_age_missing_values(df):
    """
    Replace missing age values for passengers using the average age of persons in the same passenger class.

    Presumes passenger class values = {1,2,3}
    :param df: dataframe containing samples
    :return: updated dataframe
    """
    p1_ages = df[df['Pclass'] == 1]['Age']
    p1_mean = np.mean(p1_ages)
    p2_ages = df[df['Pclass'] == 2]['Age']
    p2_mean = np.mean(p2_ages)
    p3_ages = df[df['Pclass'] == 3]['Age']
    p3_mean = np.mean(p3_ages)

    df.loc[(df.Age.isnull() & (df.Pclass == 1)), 'Age'] = p1_mean
    df.loc[(df.Age.isnull() & (df.Pclass == 2)), 'Age'] = p2_mean
    df.loc[(df.Age.isnull() & (df.Pclass == 3)), 'Age'] = p3_mean

    return df


def handle_fare_missing_values(df):
    p1_fares = df[df['Pclass'] == 1]['Fare']
    p1_mean = np.mean(p1_fares)
    p2_fares = df[df['Pclass'] == 2]['Fare']
    p2_mean = np.mean(p2_fares)
    p3_fares = df[df['Pclass'] == 3]['Fare']
    p3_mean = np.mean(p3_fares)

    df.loc[(df.Fare.isnull() & (df.Pclass == 1)), 'Fare'] = p1_mean
    df.loc[(df.Fare.isnull() & (df.Pclass == 2)), 'Fare'] = p2_mean
    df.loc[(df.Fare.isnull() & (df.Pclass == 3)), 'Fare'] = p3_mean

    return df


def handle_cabin_missing_values(df):

    def truncate_cabin_name(x):
        try:
            return x[0]
        except TypeError:
            return 'None'

    df['Cabin'] = df.Cabin.apply(truncate_cabin_name)
    return df


def handle_sibsp_missing_values(df):
    df.loc[df.SibSp.isnull(), 'SibSp'] = 0
    return df


def handle_parch_missing_values(df):
    df.loc[df.Parch.isnull(), 'Parch'] = 0
    return df


def handle_missing_features(missing_features, df):
    if len(missing_features) == 0:
        return df

    feature_handlers = {'Age': handle_age_missing_values,
                        'Fare': handle_fare_missing_values,
                        'Embarked': handle_embark_missing_values,
                        'Cabin': handle_cabin_missing_values,
                        'SibSp': handle_sibsp_missing_values,
                        'Parch': handle_parch_missing_values}

    for feature in missing_features:
        print("Feature missing data:", feature)
        if feature in feature_handlers:
            handler = feature_handlers[feature]
            df = handler(df)
    return df


def compute_accuracy(tn,tp, observation_count):
    return (tn+tp)/float(observation_count)


def process_training_data(df_train):
    missing_features = get_missing_feature_list(df_train)
    df_train = handle_missing_features(missing_features, df_train)
    # replace Sex attribute with numeric encoding
    df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
    y_predict = df_train['Survived']

    # The family feature did not result in a better training score
    df_train['Family'] = df_train['SibSp'] + df_train['Parch']
    # Drop PassengerId, Survived, Name and Ticket features for use in model
    predict_features = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)

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
    training_accuracy = compute_accuracy(tn, tp, val_X.shape[0])
    print("Training accuracy =", training_accuracy)

    return rf_classifier, list(predict_features)


def process_test_data(df_test, rf_classifier, training_features):
    # Predict survivors using test data
    missing_features = get_missing_feature_list(df_test)

    # Drop PassengerId, Name and Ticket features for use in model

    df_test = handle_missing_features(missing_features, df_test)
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
    df_train, df_test = read_data_files()
    rf_classifier, training_features = process_training_data(df_train)
    # Predict survivors using test data
    # process_test_data(df_test, rf_classifier, training_features)


if __name__ == "__main__":
    run_main()
