import os
import pandas as pd



def main():
    for fn in os.listdir("./match-data"):
        dataset = pd.read_csv(fn)




if __name__ == "__main__":
    main()