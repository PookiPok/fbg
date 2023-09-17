import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import utils
import argparse

class ModelClassifier:
    def __init__(self, layer=15, activation='relu', solver='lbfgs',verbose=True):
        self.layer = layer
        self.model = MLPClassifier(hidden_layer_sizes=(layer,), activation=activation, solver=solver, alpha=0.0001, max_iter=700, verbose=verbose)


def trained_builder(model,dataset_csv, trained_file, only_test=False):

    X,y = utils.readCSV(dataset_csv)
    if only_test == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model.model.fit(X_train, y_train)
        y_pred = model.model.predict(X_test)
        accuracy = model.model.score(X_test, y_test)
        print("Accuracy for layer:" + str(model.layer) + ' is:' + str(accuracy))
        joblib.dump(model.model, trained_file + '_' + str(round(accuracy, 3)) + '_' + str(model.layer)+'.pkl')
    else:
        loaded_model = joblib.load(trained_file)
        y_pred = loaded_model.predict(X)
        print(classification_report(y, y_pred))
        print('Accuracy: %.3f' % accuracy_score(y, y_pred) + ' for layer:' + str(model.layer))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True,type=str,help="Dataset for build the model")
    ap.add_argument("-m", "--model", required=True,help="Model file name")
    ap.add_argument("-t", "--only_test", action="store_true", required=False,help="build on new model without testing it")
    ap.add_argument("-l", "--layer", default=15,type=int, required=False,help="How many layers to train (default=15)")
    ap.add_argument("-a", "--activation", default="relu",type=str, required=False,help="relu,identity, logistic, tanh")
    ap.add_argument("-s", "--solver", default="lbfgs",type=str, required=False,help="lbfgs,sgd,adam")
    ap.add_argument("-v", "--verbose", default=True,type=bool, required=False,help="build on new model without testing it")
    args = ap.parse_args()
    model = ModelClassifier(args.layer, args.activation, args.solver, args.verbose)
    trained_builder(model, args.dataset, args.model, args.only_test)