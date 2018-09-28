import pandas as pd
import numpy as np


def compute_accuracy(tn,tp, observation_count):
    return (tn+tp)/float(observation_count)


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


def handle_missing_features(missing_features, df):
    if len(missing_features) == 0:
        return df

    feature_handlers = {'Age': handle_age_missing_values,
                        'Fare': handle_fare_missing_values,
                        'Embarked': handle_embark_missing_values,
                        'Cabin': handle_cabin_missing_values}

    for feature in missing_features:
        print("Feature missing data:", feature)
        if feature in feature_handlers:
            handler = feature_handlers[feature]
            df = handler(df)
    return df


def join_feature_name_with_importance_value(features, importances):
    '''
    Join via a list of tuples, feature names with their importance values
    :param features: data frame whose features are represented by columns used by classifier
    :param importances: feature importance scores assigned by classifier
    :return: sorted list (highest importances first) of feature,importance tuples
    '''
    if features.columns.shape[0] != importances.shape[0]:
        return []

    feature_importances = []
    for item in range(features.columns.shape[0]):
        feature_importances.append((features.columns[item], importances[item]))
    feature_importances_sorted = sorted(feature_importances, reverse=True, key=lambda kv: kv[1])

    return feature_importances_sorted

