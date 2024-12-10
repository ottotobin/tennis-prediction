import os
import pandas as pd

def process(dataset):
    



def main():
    for fn in os.listdir("./match-data"):
        dataset = pd.read_csv(fn)
        process(dataset)




if __name__ == "__main__":
    main()