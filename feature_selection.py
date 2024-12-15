import argparse
import pandas as pd
from models import merge_data, convert_labels, split_data, convert_labels, create_validation, create_network, train_network, calculate_accuracy
from sklearn import feature_selection, linear_model

#----GLOBALS----
TRAIN_PERCENT = 0.80
SEED = 2200

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

    # converting new training and testing into dataframes
    select_training_X = pd.DataFrame(data=new_training_X, columns=selected_attributes)
    select_testing_X = pd.DataFrame(data=new_testing_X, columns=selected_attributes)

    # return the transformed training and testing set instances
    return select_training_X, select_testing_X, selected_attributes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    df = merge_data()
    dataset = convert_labels(df)

    training_X, training_y, testing_X, testing_y = split_data(dataset, TRAIN_PERCENT, SEED)

    highest_accuracy = 0
    for k in range(1, training_X.shape[1] + 1):
        select_train_X, select_test_X, selected_attributes = selection(training_X, training_y, testing_X, k)

        if args.model == "neural":
            inputs = pd.concat([select_train_X, training_y])
            network = create_network(inputs, 128)
            valid_accuracy = train_network(network, select_train_X, training_y, 0.01)
            print((k, valid_accuracy, selected_attributes))
        else:
            regressor = linear_model.LogisticRegression()
            regressor.fit(select_train_X, training_y)
            valid_accuracy = calculate_accuracy(regressor, select_train_X, training_y)
            print((k, valid_accuracy, selected_attributes))

        if valid_accuracy > highest_accuracy:
            highest_accuracy = valid_accuracy
            best_attributes = (k, highest_accuracy, selected_attributes)

    print("Best k, accuracy, and selected attributes")
    print(best_attributes)

        
if __name__ == "__main__":
    main()

