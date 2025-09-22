import os
import pandas as pd

def game_diff(score_str):
    diff = 0
    for set in score_str.split(" "):
        games = set.split("-")
        won1 = int(games[0])
        won2 = int(games[1])

        diff += abs(won1 - won2)

    return diff

def avg_margins(matches):
    matches["score_diff"] = matches.apply(lambda x: game_diff(x["score"]))
    pd.to_csv("match_margins.csv")
    return matches["score_diff"].mean()

def main():
    scores_frames = []
    for matchfile in os.listdir("./raw-matchdata"):
        season = pd.read_csv("./raw-matchdata/"+matchfile)
        scores = season["score"]
        scores_frames.append(scores)

    all_scores = pd.concat(scores_frames)
    avg_margins(all_scores)

if __name__ == "__main__":
    main()