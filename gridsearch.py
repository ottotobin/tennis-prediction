import pandas as pd
import os
from models import merge_data, split_data, convert_labels, create_network, train_network


#----GLOBALS----
TRAIN_PERCENT = 0.80
SEED = 2200

# creates a row in a hyperparam_tuning.csv file for each run 
# of a specificied model
def results(rate, neurons, performance):
    row = pd.DataFrame(data={
        "Rate": [rate],
        "Neurons": [neurons],
        "Performance": [performance]
    })
    
    if os.path.exists("hyperparam_tuning.csv"):
        row.to_csv("hyperparam_tuning.csv", mode="a", float_format="%.4f", header=False, index=False)
    else:
        row.to_csv("hyperparam_tuning.csv", float_format="%.4f", header=True, index=False)


def main():
    df = merge_data()
    dataset = convert_labels(df)

    training_X, training_y, testing_X, testing_y = split_data(dataset, TRAIN_PERCENT, SEED)

    for neurons in [32, 64, 128, 256, 320]:
        for lr in [0.0001, 0.001, 0.01, 0.1]:
            network = create_network(dataset, neurons)
            valid_performance = train_network(network, training_X, training_y, lr)
            results(lr, neurons, valid_performance)

    


if __name__ == "__main__":
    main()