import os
import math
import pandas as pd
from sklearn import linear_model, feature_selection
import torch
from plotnine import ggplot, aes, geom_col


#----GLOBALS----
SEED = 2200
TRAIN_PERCENT = 0.80
NEURONS = 32
RATE = 0.01

#----PROCESSING----

# merge preprocessed matchfiles into one big df
def merge_data():
    frames = []
    for matchfile in os.listdir("./processed-matchdata"):
        dataset = pd.read_csv("./processed-matchdata/"+matchfile)
        frames.append(dataset)

    df = pd.concat(frames).drop("Unnamed: 0", axis=1)
    return df.dropna()

# split dataset into training and testing
def split_data(dataset, train_percentage, seed):
    data = dataset.drop(["player1", "player2"], axis=1)
    shuffled = data.sample(frac=1, random_state=seed)
    total_rows = shuffled.shape[0]
    training_rows = int(train_percentage * total_rows)
    
    training = shuffled.iloc[:training_rows, :]
    testing = shuffled.iloc[training_rows:, :]
    
    training_X = training.drop("label", axis=1)
    training_y = training["label"]
    
    testing_X = testing.drop("label", axis=1)
    testing_y = testing["label"]
    
    return training_X, training_y, testing_X, testing_y

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

#----LOGISTIC REGRESSION----
def train_log_regression(training_X, training_y):
    model = linear_model.LogisticRegression()
    model.fit(training_X, training_y)
    
    return model

#----DEEP LEARNING----

# converts the the labels of mnist1000 into numbers
def convert_labels(dataset):
    dataset_processed = dataset.copy()
    # create a dictionary mapping
    map = {"player1": 1, "player2": 0}
    
    # convert each of the string labels into the corresponding number
    for name in map:
        number = map[name]
        dataset_processed.loc[dataset["label"] == name, "label"] = number

    # make sure the column type is a number
    dataset_processed["label"] = pd.to_numeric(dataset_processed["label"])
    return dataset_processed

# creates a neural network with one hidden layer with a given number of attributes and labels
def create_network(dataset, neurons):
    # calculate inputs
    inputs = dataset.shape[1] - 1

    # this creates a hidden layer w/ given number of neurons and inputs for each instance
    hidden_layer = [
        torch.nn.Linear(inputs, neurons),
        torch.nn.Sigmoid(),
    ]

    # calculate outputs if classification
    outputs = len(dataset["label"].unique())
    output_layer = [
        torch.nn.Linear(neurons, outputs),
        torch.nn.Softmax(dim=1),
    ]
            
    # combine all the layers into a single list
    all_layers = hidden_layer + output_layer

    # turn the layers into a neural network
    network = torch.nn.Sequential(*all_layers)

    return network

# converts a training set into smaller train and validation sets
def create_validation(training_X, training_y, valid_percentage):
    # find the split point between training and validation
    training_n = training_X.shape[0]
    valid_rows = int(valid_percentage * training_n)

    # create the validation set
    valid_X = training_X.iloc[:valid_rows]
    valid_y = training_y.iloc[:valid_rows]

    # create the (smaller) training set
    train_X = training_X.iloc[valid_rows:]
    train_y = training_y.iloc[valid_rows:]

    return train_X, train_y, valid_X, valid_y

# trains a neural network with given training data
def train_network(network, training_X, training_y, rate, verbose=False):
    # split the training data into train and validation
    # Note: use 20% of the original training data for validation
    train_X, train_y, valid_X, valid_y = create_validation(training_X, training_y, 0.2)
    
    # convert our data to PyTorch objects
    train_X = torch.from_numpy(train_X.values).float()
    train_y = torch.from_numpy(train_y.values).long()
    valid_X = torch.from_numpy(valid_X.values).float()
    valid_y = torch.from_numpy(valid_y.values).long()

    # move the data and model to the GPU if possible
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        train_X = train_X.to(device)
        train_y = train_y.to(device)
        valid_X = valid_X.to(device)
        valid_y = valid_y.to(device)      
        network = network.to(device)
    

    # create the algorithm that learns the weight for the network (with a learning rate of 0.01)
    optimizer = torch.optim.Adam(network.parameters(), lr=rate)

    # create the loss function function that tells optimizer how much error it has in its predictions
    # here we use cross entropy since we have a classification task with more than two possible labels
    loss_function = torch.nn.CrossEntropyLoss()

    valid_acc_values = []
    
    # train for 1000 epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # make predictions on the training set and validation set
        train_predictions = network(train_X)
        valid_predictions = network(valid_X)

        # calculate the error on the training set
        train_loss = loss_function(train_predictions, train_y)
        
        # # calculate accuracy for validation sets
        valid_accuracy = calculate_accuracy(network, valid_X, valid_y)

        # perform backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        valid_acc_values.append(valid_accuracy)

        # stopping training early if reach a perfect validation accuracy
        if valid_accuracy[0] >= 1:
            break
    
    return valid_acc_values[-1]


#----PERFORMANCE----

# calculates the accuracy of the predictions of a given classification model
def calculate_accuracy(model, X, y):
    if isinstance(model, torch.nn.modules.container.Sequential):
        # make predictions for the given X
        if not isinstance(X, torch.Tensor) and not isinstance(y, torch.Tensor):
            X = torch.from_numpy(X.values).float()
            y = torch.from_numpy(y.values).long()

        if torch.cuda.is_available():
            device = torch.device('cuda')

            X = X.to(device)
            y = y.to(device)
            model = model.to(device)

        softmax_probs = model(X)    
        predictions = torch.argmax(softmax_probs, dim=1)

        # calculate accuracy of those predictions
        accuracy = sum(predictions == y) / len(predictions)
    else:
        predictions = model.predict(X)
        correct = sum(predictions == y)
        accuracy = correct / len(X)
    
    return float(accuracy), predictions

# calculates the confidence interval for a given model's performance
def confidence_interval(performance, predictions):
    z = 1.96
    n = len(predictions)
    standard_error = math.sqrt((performance*(1-performance)) / n)
    interval = (performance - z*standard_error, performance + z*standard_error)

    return interval    

#---VISUALIZATION---
def create_barchart(df):
    plot = (
        ggplot(df)
        + aes(x="model", y="accuracy", fill="model")
        + geom_col()
    )
    fn = "barchart_accuracy.png"
    plot.save(filename=fn)


def main():
    df = merge_data()
    dataset = convert_labels(df)

    # split data and then do feature selection
    training_X, training_y, testing_X, testing_y  = split_data(dataset, TRAIN_PERCENT, SEED)
    selected_train_X_logistic, selected_test_X_logistic, selected_logistic = selection(training_X, training_y, testing_X, 3)
    selected_train_X_neural, selected_test_X_neural, selected_neural = selection(training_X, training_y, testing_X, 13)

    # train Logistic Regression Model 
    log_reg = train_log_regression(selected_train_X_logistic, training_y)

    # train Neural Network
    inputs = pd.concat([selected_train_X_neural, training_y])

    network = create_network(inputs, NEURONS)
    train_network(network, selected_train_X_neural, training_y, RATE)

    # calculate accuracy
    logistic_accuracy, logistic_predictions = calculate_accuracy(log_reg, selected_test_X_logistic, testing_y)
    neural_accuracy, neural_predictions = calculate_accuracy(network, selected_test_X_neural, testing_y)

    # calculate confidence intervals
    logistic_interval = confidence_interval(logistic_accuracy, logistic_predictions)
    neural_interval = confidence_interval(neural_accuracy, neural_predictions)

    print("logistic confidence interval", logistic_interval)
    print("neural confidence interval", neural_interval)


    # plot accuracy comparison
    model_accuracies = {"model": ["Logistic Regression", "Neural Network"], "accuracy": [logistic_accuracy, neural_accuracy]}
    create_barchart(pd.DataFrame(data=model_accuracies))

if __name__ == "__main__":
    main()