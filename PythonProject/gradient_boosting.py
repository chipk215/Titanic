
import common
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix


def process_training_data(df_train):
    missing_features = common.get_missing_feature_list(df_train)
    df_train = common.handle_missing_features(missing_features, df_train)
    # replace Sex attribute with numeric encoding
    df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
    y_predict = df_train['Survived']

    # Drop PassengerId, Survived, Name and Ticket features for use in model
    predict_features = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)

    train_X, val_X, train_y, val_y = train_test_split(predict_features, y_predict,  random_state=0)
    tree_count = 1000
    classifier = GradientBoostingClassifier(n_estimators=tree_count, learning_rate=1.0, max_depth=1, random_state=0)
    classifier.fit(train_X, train_y)
    score = classifier.score(train_X, train_y)
    print("Model score= ", score)
    survivor_predictions = classifier.predict(val_X)
    print(confusion_matrix(val_y, survivor_predictions))
    tn, fp, fn, tp = confusion_matrix(val_y, survivor_predictions).ravel()
    training_accuracy = common.compute_accuracy(tn, tp, val_X.shape[0])
    print("Training accuracy =", training_accuracy)

    training_records = df_train.loc[val_X.index.values]
    passengerIds = training_records['PassengerId']

    result = pd.DataFrame()
    result['PassengerId'] = passengerIds
    result['Survived'] = val_y
    result['Prediction'] = survivor_predictions
    return classifier, list(predict_features),  result


