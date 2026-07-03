import argparse
import joblib
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)

from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline

# Scalers and classifiers from scikit-learn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import (
    MaxAbsScaler,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

"""
Other classifiers and scalers can be added to the dictionaries
(need to add the corresponding import):
    - Key of the dictionary: used in the command line
    - Value: classifier/scaler class from scikit-learn
"""

classifiers = {
    "dec_tree": DecisionTreeClassifier(
        max_depth=9,
        min_samples_split=2,
        min_samples_leaf=3,
        splitter="best",
        random_state=0,
    ),
    "rand_for": RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2, random_state=0
    ),
    "svm": SVC(gamma="auto", kernel="linear", C=3, probability=True),
    "kneighbors": KNeighborsClassifier(n_neighbors=2, leaf_size=3, weights="distance"),
    "logistic": LogisticRegression(max_iter=10000, random_state=0),
    "neural": MLPClassifier(
        hidden_layer_sizes=(200,), alpha=0.01, max_iter=1000, random_state=0
    ),
    "dummy": DummyClassifier(strategy="stratified", random_state=0),
}
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "maxabs": MaxAbsScaler(),
    "robust": RobustScaler(quantile_range=(25, 75)),
}

# Name of the file in which the trained model is saved
file_model = "./results/model.sav"

# Name of the column with the labels
target = "_Target"

# Order of the targets for the confusion matrix
labels = [
    "SLG",
    "0-9°",
    "9-20°",
    "20-30°",
]

# Shuffle (True) or not (False) the training deck
shuffle_train = True


def main():

    parser = argparse.ArgumentParser(
        description="Command-line arguments", fromfile_prefix_chars="@"
    )

    subparsers = parser.add_subparsers(dest="subs", help="Sub-command help")

    # Parser for training
    parser_train = subparsers.add_parser("train", help="Training")
    parser_train.add_argument(
        "train_file",
        metavar="TRAINING_FILE",
        help="File with the training dataset",
    )
    parser_train.add_argument(
        "model",
        choices=list(classifiers.keys()),
        help="Model used for the training",
    )
    parser_train.add_argument(
        "-s",
        "--scaler",
        choices=list(scalers.keys()),
        help="Scaler",
    )
    parser_train.add_argument(
        "-p",
        "--predict",
        dest="predict_file",
        metavar="PREDICT_FILE",
        help="If provided, predict a given file immediately after the training",
    )

    # Parser for predictions
    parser_predict = subparsers.add_parser("predict", help="Predicting")
    parser_predict.add_argument(
        "trained_model",
        metavar="TRAINED_MODEL",
        help="File with the model trained",
    )
    parser_predict.add_argument(
        "predict_file",
        metavar="PREDICT_FILE",
        help="File to be predicted",
    )

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    if args.subs == "train":
        training(args.train_file, args.model, args.scaler, args.predict_file)
    elif args.subs == "predict":
        predicting(trained_model=args.trained_model, predict_file=args.predict_file)



def training(train_file, model, scaler=None, predict=False, split=True, crossval=True):

    print("TRAINING")
    print(f"Training file: {train_file}")
    print(f"Training model {model}")
    print(f"Scaler: {scaler}")

    trainset = _load_csv(train_file)

    if trainset is None:
        print(f"{train_file} is not a valid training file or doesn't exist")
        return False

    # Separates the features and the target from the trainset
    X = trainset.drop(target, axis=1)
    y = trainset[target]

    # Shuffle the train set deck
    if shuffle_train:
        X, y = shuffle(X, y, random_state=0)

    # Split the deck into train and test decks

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    model = classifiers[model]

    if scaler:
        scaler = scalers[scaler]

    # Construct the pipeline
    pipe = make_pipeline(scaler, model)

    # Performs the cross validation
    if crossval:
        cross_validation(X_train, y_train, pipe, shuffle_train)

    # Trains the model
    pipe.fit(X_train, y_train)

    # Checking the accuracy with the (hold-out) test set
    if split:
        test_result = pipe.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, test_result)
        print(f"\nAccuracy on the testset: {accuracy}\n")

    # Save the trained model
    joblib.dump(pipe, file_model)

    # Call the predicting function
    if predict:
        predicting(trained_model=file_model, predict_file=predict)


def cross_validation(X, y, model, shuffle_train=True):

    print("\nCROSS-VALIDATION")
    if shuffle_train == False:
        rand_state = None
    else:
        rand_state = 0
    cv = StratifiedKFold(n_splits=5, shuffle=shuffle_train, random_state=rand_state)

    results = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

    print(f"Cross-validation scores: {results}")

    y_train_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    # Computing scores
    # Confusion matrix: rows = real labels, columns = predicted labels
    conf_matrix = metrics.confusion_matrix(
        y, y_train_pred, labels=labels, normalize=None
    )

    norm_conf = np.sum(conf_matrix, axis=1)
    norm_conf = norm_conf.repeat(4).reshape(4, 4)
    conf_matrix = conf_matrix/norm_conf
    conf_matrix = np.around(conf_matrix,decimals = 2)
    plt.imshow(conf_matrix, cmap=plt.cm.Reds)
    plt.colorbar()
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.annotate(conf_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.xticks(range(0, 4), labels=labels)  # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.yticks(range(0, 4), labels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

    conf_matrix = pd.DataFrame(data=conf_matrix, columns=labels, index=labels)

    # Accuracy
    accuracy = results.mean()
    std_dev = results.std()
    print(f"Cross-validation accuracy: {accuracy} ± {std_dev}")

    precision, sensitivity, fbscore, support = metrics.precision_recall_fscore_support(
        y, y_train_pred, beta=1.0, labels=labels, average=None
    )

    model_metrics = pd.DataFrame(
        {
            "Precision": precision,
            "Sensitivity": sensitivity,
            "F-beta score": fbscore,
            "Number": support,
        },
        index=labels,
    )



    print(f"\nConfusion matrix:\n{conf_matrix}")
    conf_matrix.to_csv("results/confusion_matrix.csv")
    print(f"\n Model metrics:\n{model_metrics}")
    model_metrics.to_csv("results/model_metrics.csv")



def predicting(trained_model, predict_file):
    print("PREDICTING")
    print(f"Predict file: {predict_file}")
    print(f"Trained model {trained_model}")

    predict = _load_csv(predict_file)
    model = joblib.load(trained_model)

    # Removes the data that could not be manually determined
    predict = predict[~predict[target].isin(["NotDetermined"])]

    # Separates the test data in features (X) and labels (y, if present)
    if target in predict.columns:
        X_predict = predict.drop(target, axis=1)
        y_predict = predict[target]
    else:
        X_predict = predict
        y_predict = None

    # Performs the prediction
    predict_result = model.predict(X_predict)

    if target in predict.columns:
        accuracy = metrics.accuracy_score(y_predict, predict_result)
        print(f"Prediction accuracy: {accuracy}")

        y_predict = pd.DataFrame(
            y_predict.values.tolist(), columns=["Real value"], index=y_predict.index
        )
        y_predict["Prediction"] = predict_result.tolist()
    else:
        y_predict = pd.DataFrame(predict_result, columns="Prediction")

    # Saves the prediction in a file
    y_predict.to_csv("results/prediction.csv")


def _load_csv(file):
    """Open a csv file and stores it as a pandas dataframe"""

    try:
        with open(file) as f:
            df = pd.read_csv(f, sep=None, engine="python")
            return df
    except Exception as err:
        print(f"Error loading .csv file: {file}")
        print(err)
        sys.exit()


if __name__ == "__main__":
    main()
