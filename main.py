from bayesian_classifier import BayesianClassifier

import pandas as pd
import csv
import string


def process_data(data_file):
    """
    Function for data processing and returns DataFrame.
    :param data_file: str - path to train data
    :return: pd.DataFrame|list - id, messages and authors data frames
    """
    stop_words = read_stop_words()
    data = pd.read_csv(data_file)

    lst_for_rep = []
    for index, i in data[['text']].iterrows():
        t = i['text'].lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
        for j in t:
            if j not in stop_words:
                lst_for_rep.append(j)
        data.at[index, 'text'] = lst_for_rep
        lst_for_rep = []
    return data


def read_stop_words():
    """
    Reads file and returns list with words from file.
    """
    lst = []
    with open('authors/stop_words.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lst += row
    return lst


if __name__ == "__main__":
    train = process_data("authors/train.csv")
    test = process_data("authors/test.csv")

    classifier = BayesianClassifier()
    classifier.fit(train)
    print(f"Model score: {classifier.score(test)}%")
