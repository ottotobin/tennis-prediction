import os
import pandas as pd

def clean(dataset):
    cleaned = dataset.copy()

    # get just attributes need for models
    cleaned = cleaned[["Tourney_name", "Surface", "Tourney_level", "Score", "Best_of", 
    "Winner_name", "W_ace", "W_df", "W_svpt", "W_1stIn", "W_1stWon", "W_2ndWon", "W_bpSaved", "W_bpFaced",
    "Loser_name", "L_ace", "L_df", "L_svpt", "L_1stIn", "L_1stWon", "L_2ndWon", "L_bpSaved", "L_bpFaced"]]

    

    



def main():
    for fn in os.listdir("./match-data"):
        dataset = pd.read_csv(fn)
        clean()




if __name__ == "__main__":
    main()