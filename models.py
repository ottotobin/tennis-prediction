import os
import pandas as pd
from sklearn import linear_model
import torch


#----GLOBALS----
SEED = 2200
TRAIN_PERCENT = 0.80
NEURONS = 128
RATE = 0.01

#----PROCESSING----

# merge preprocessed matchfiles into one big df
def merge_data():
    frames = []
    for matchfile in os.listdir("./processed-matchdata"):
        dataset = pd.read_csv("./processed-matchdata/"+matchfile)
        frames.append(dataset)

    df = pd.concat(frames).drop("Unnamed: 0", axis=1)
    return df

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


# creates a neural network with one hidden layer with a given number of attributes and labels
def create_network(dataset, neurons):
    # calculate inputs
    inputs = dataset.drop(["player1", "player2"], axis=1).shape[1] - 1

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

    # dataframe columns
    epochs = []
    train_loss_values = []
    valid_loss_values = []
    train_acc_values = []
    valid_acc_values = []
    
    # train for 1000 epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # make predictions on the training set and validation set
        train_predictions = network(train_X)
        valid_predictions = network(valid_X)

        # calculate the error on the training set
        train_loss = loss_function(train_predictions, train_y)
        train_error = train_loss.item()
                
        # calculate error on the validation set
        valid_loss = loss_function(valid_predictions, valid_y)
        valid_error = valid_loss.item()
        
        # calculate accuracy for both training and validation sets
        train_accuracy = calculate_accuracy(network, train_X, train_y)
        valid_accuracy = calculate_accuracy(network, valid_X, valid_y)

        # perform backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # saving error and accuracy to a dataframe
        epochs.append(epoch)
        train_loss_values.append(train_error)
        valid_loss_values.append(valid_error)
        train_acc_values.append(train_accuracy)
        valid_acc_values.append(valid_accuracy)
    
    # convert the training progress data to a Pandas DataFrame
    progress = {
        "epoch": epochs,
        "train_loss": train_loss_values,
        "valid_loss": valid_loss_values,
        "train_acc": train_acc_values,
        "valid_acc": valid_acc_values
    }
    
    return pd.DataFrame(progress)

#----PERFORMANCE----

# calculates the accuracy of the predictions of a given classification model
def calculate_accuracy(model, X, y):
    if isinstance(model, torch.nn.modules.container.Sequential):
        # make predictions for the given X
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
    
    return float(accuracy)

def main():
    df = merge_data()
    # dataset.to_csv("matches2010-20.csv")
    dataset = convert_labels(df)

    training_X, training_y, testing_X, testing_y  = split_data(dataset, TRAIN_PERCENT, SEED)
    test_X = testing_X.to_numpy()
    test_y = testing_y.to_numpy()

    # log_reg = train_log_regression(training_X, training_y)

    network = create_network(dataset, NEURONS)
    train_network(network, training_X, training_y, RATE)

    print("neural network accuracy:", calculate_accuracy(network, test_X, test_y))


if __name__ == "__main__":
    main()