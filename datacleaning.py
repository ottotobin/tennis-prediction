import os
import pandas as pd
import random

ATTRIBUTES = ["surface", "tourney_level", "best_of", 
    "winner_name", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "loser_name", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]

W_STATS = ["w_ace", "w_df", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved"]
L_STATS = ["l_ace", "l_df", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved"]

# creates a percentage if there exists a total
def create_percentage(count, total):
    if total != 0:
        return count / total
    else:
        return 1

def assign_players(winner, loser):
    rand = random.random()
    if rand < 0.5:
        return winner
    else:
        return loser

def create_feature(player1, winner, w_percent, l_percent):
    if player1 == winner:
        return w_percent - l_percent
    else:
        return l_percent - w_percent

# labels the match based on who won
def create_label(player1, player2, winner):
    if player1 == winner:
        return "player1"
    else:
        return "player2"

# hot-encodes categorical variables
def encode(dataset):
    dataset["best_of"] = dataset["best_of"].astype(object, copy=True)
    categoricals = [column for column in dataset.select_dtypes(include=["object"]).columns if column not in ["winner_name", "loser_name", "player1", "player2", "label"]]
    for column in dataset.columns:
        if column in categoricals:    
            onehots = pd.get_dummies(dataset[column], column, drop_first=False, dtype=int)
            dataset = pd.concat([dataset.drop(column, axis=1), onehots], axis=1)

    return dataset

def clean(dataset):
    cleaned = dataset.copy()

    # get just attributes needed for models
    cleaned = cleaned[ATTRIBUTES]

    # get just ATP matches at Grand Slam(G), Masters 1000(M), and Tour-level(A)
    cleaned = cleaned[cleaned["tourney_level"].isin(["G", "M", "A"])]

    # create percentages for each winner and loser
    for w_count, l_count in zip(W_STATS, L_STATS):
        if "bp" in w_count:
            w_total = "w_bpFaced"
            l_total = "l_bpFaced"
        else:
            w_total = "w_svpt"
            l_total = "l_svpt"
        cleaned[w_count + "%"] = cleaned.apply(lambda x: create_percentage(x[w_count], x[w_total]), axis = 1)
        cleaned[l_count + "%"] = cleaned.apply(lambda y: create_percentage(y[l_count], y[l_total]), axis=1)


    # randomly assign winner and loser as Player 1 and Player 2
    cleaned["player1"] = cleaned.apply(lambda x: assign_players(x["winner_name"], x["loser_name"]), axis=1)
    cleaned["player2"] = cleaned.apply(lambda x: x["loser_name"] if x["player1"] == x["winner_name"] else x["winner_name"], axis=1)

    # create features in the form: feature = stat_1 - stat_2
    for w_count, l_count in zip(W_STATS, L_STATS):
        w_percent = w_count + "%"
        l_percent = l_count + "%"
        feature_name = w_count[w_count.find("_") + 1:] + "%"
        cleaned[feature_name] = cleaned.apply(lambda x: create_feature(x["player1"], x["winner_name"], x[w_percent], x[l_percent]), axis=1)

    # create labels
    cleaned["label"] = cleaned.apply(lambda x: create_label(x["player1"], x["player2"], x["winner_name"]), axis=1)

    # removed winner and loser count stats
    removables = W_STATS + L_STATS + [stat + "%" for stat in W_STATS + L_STATS] + ["w_svpt", "w_bpFaced", "l_svpt", "l_bpFaced", "winner_name", "loser_name"]
    
    cleaned = cleaned.drop(removables, axis=1)

    # will pass each instance as a 3-tuple in the form: (player1, player2, instance)
    return cleaned


def main():
    print(os.listdir("./raw-matchdata"))
    for rawfile in os.listdir("./raw-matchdata"):
        matchfile = "./raw-matchdata/" + rawfile
        dataset = pd.read_csv(matchfile)

        cleaned = clean(dataset)
        encoded = encode(cleaned)
        processedfile = "./processed-matchdata/" + rawfile.strip("_")
        encoded.to_csv(processedfile)


if __name__ == "__main__":
    main()