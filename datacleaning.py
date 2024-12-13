import os
import pandas as pd
import random

ATTRIBUTES = ["tourney_name", "surface", "tourney_level", "score", "best_of", 
    "winner_name", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "loser_name", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]

W_STATS = ["w_ace", "w_df", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved"]
L_STATS = ["l_ace", "l_df", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved"]



    # cleaned["w_ace%"] = cleaned.apply(lambda x: x["w_ace"] / x["w_svpt"], axis=1)
    # cleaned["w_df%"] = cleaned.apply(lambda x: x["w_df"] / x["w_svpt"], axis=1)
    # cleaned["w_1stIn%"] = cleaned.apply(lambda x: x["w_1stIn"] / x["w_svpt"], axis=1)
    # cleaned["w_1stWon%"] = cleaned.apply(lambda x: x["w_1stWon"] / x["w_svpt"], axis=1)
    # cleaned["w_2ndWon%"] = cleaned.apply(lambda x: x["w_2ndWon"] / x["w_svpt"], axis=1)

    # cleaned["w_bpSaved%"] = cleaned.apply(lambda x: x["w_bpSaved"] / x["w_bpFaced"], axis=1)

    # # create percentages for loser
    # cleaned["l_ace%"] = cleaned.apply(lambda x: x["l_ace"] / x["l_svpt"], axis=1)
    # cleaned["l_df%"] = cleaned.apply(lambda x: x["l_df"] / x["l_svpt"], axis=1)
    # cleaned["l_1stIn%"] = cleaned.apply(lambda x: x["l_1stIn"] / x["l_svpt"], axis=1)
    # cleaned["l_1stWon%"] = cleaned.apply(lambda x: x["l_1stWon"] / x["l_svpt"], axis=1)
    # cleaned["l_2ndWon%"] = cleaned.apply(lambda x: x["l_2ndWon"] / x["l_svpt"], axis=1)

    # cleaned["l_bpSaved%"] = cleaned.apply(lambda x: x["l_bpSaved"] / x["l_bpFaced"], axis=1)

# creates a percentage if there exists a total
def create_percentage(count, total):
    if total != 0:
        return count / total
    else:
        return 1

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

def clean(dataset):
    cleaned = dataset.copy()

    # get just attributes needed for models
    cleaned = cleaned[ATTRIBUTES]

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
    cleaned["player1"] = [None for _ in range(cleaned.shape[0])]
    cleaned["player2"] = [None for _ in range(cleaned.shape[0])]
    for index, row in cleaned.iterrows():
        rand = random.random()
        print(rand)
        if rand < 0.5:
            print("player1 is winner")
            row["player1"] = row["winner_name"]
            row["player2"] = row["loser_name"]
        else:
            print("player2 is winner")
            row["player2"] = row["winner_name"]
            row["player1"] = row["loser_name"]

    print(cleaned.head())
    
    # create features in the form: feature = stat_1 - stat_2
    for w_count, l_count in zip(W_STATS, L_STATS):
        w_percent = w_count + "%"
        l_percent = l_count + "%"
        feature_name = w_count[w_count.find("_"):] + "%"
        cleaned[feature_name] = cleaned.apply(lambda x: create_feature(x["player1"], x["winner_name"], x[w_percent], x[l_percent]), axis=1)

    # create labels
    cleaned["label"] = cleaned.apply(lambda x: create_label(x["player1"], x["player2"], x["winner_name"]), axis=1)

    # will pass each instance as a 3-tuple in the form: (player1, player2, instance)
    print(cleaned.head())
    return cleaned

# process
#   encode surface
#   encode best_of
#   encode tourney_level




def main():
    # for fn in os.listdir("./raw-matchdata"):
    #     dataset = pd.read_csv(fn)
    #     clean(dataset)

    dataset = pd.read_csv("./raw-matchdata/atp_matches_2013.csv")
    # print(dataset.iloc[0])
    cleaned = clean(dataset)


if __name__ == "__main__":
    main()