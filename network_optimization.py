import pandas as pd
from sklearn import feature_selection
import os
from models import merge_data, split_data, convert_labels, create_network, train_network


#----GLOBALS----
TRAIN_PERCENT = 0.80
SEED = 2200

# creates a row in a hyperparam_tuning.csv file for each run 
# of a specificied model
def results(attributes, rate, neurons, performance):
    row = pd.DataFrame(data={
        "Attributes": [attributes],
        "Rate": [rate],
        "Neurons": [neurons],
        "Performance": [performance]
    })
    
    if os.path.exists("hyperparam_tuning.csv"):
        row.to_csv("hyperparam_tuning.csv", mode="a", float_format="%.4f", header=False, index=False)
    else:
        row.to_csv("hyperparam_tuning.csv", float_format="%.4f", header=True, index=False)

def selection(training_X, training_y, testing_X, k=None):
    # create the feature selection algorithm
    feature_selector = feature_selection.SelectKBest(feature_selection.f_regression, k=k)

    # determine the best attributes to keep using only the training set 
    # (since the testing set wouldn't be available during training)
    new_training_X = feature_selector.fit_transform(training_X, training_y)

    # keep only the selected attributes in the testing set
    new_testing_X = feature_selector.transform(testing_X)

    # determine the attributes chosen by the algorithm:
    # chosen is a list of True/False values, one per original attribute, where True indicates the original attribute was selected
    scores = zip(feature_selector.scores_, feature_selector.feature_names_in_)
    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    chosen = [col in [score[1] for score in scores_sorted[:k]] for col in training_X.columns]

    selected_attributes = [training_X.columns[i] for i in range(len(training_X.columns)) if chosen[i]]

    # return the transformed training and testing set instances
    return new_training_X, new_testing_X, selected_attributes


def main():
    df = merge_data()
    dataset = convert_labels(df)

    training_X, training_y, testing_X, testing_y = split_data(dataset, TRAIN_PERCENT, SEED)

    for neurons in [32, 64, 128, 256, 320]:
        for lr in [0.0001, 0.001, 0.01, 0.1]:
            for k in range(1, training_X.shape[1] + 1):
                select_train_X, select_test_X, selected_attributes = selection(training_X, training_y, testing_X, k)

                network = create_network(dataset, neurons)
                valid_performance = train_network(network, select_train_X, training_y, lr)
                results(selected_attributes, lr, neurons, valid_performance)


if __name__ == "__main__":
    main()
