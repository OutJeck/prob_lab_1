from bayesian_classifier import BayesianClassifier

import pandas as pd
import csv
import string


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    stop_words = read_stop_words()
    data = pd.read_csv(data_file)

    # for column in data[['text']]:
    #     # Select column contents by column name using [] operator
    #     columnSeries = data[column]
    #     print(type(columnSeries.values))

    lst_for_rep = []  # list for the replacing old text into the new one
    for index, i in data[['text']].iterrows():
        # print(i['text'])
        t = i['text'].lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
        for j in t:
            if j not in stop_words:
                lst_for_rep.append(j)
        # print(lst_for_rep)
        data.at[index, 'text'] = lst_for_rep
        lst_for_rep = []

    return data


def read_stop_words():
    """
    Reads file and returns list with words from file.
    """
    lst = []
    with open('stop_words.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lst += row
    return lst


if __name__ == "__main__":
    # train_X, train_y = process_data("")
    # test_X, test_y = process_data("your test data file")
    print(process_data("data/0-authors/train.csv"))


    # classifier = BayesianClassifier()
    # classifier.fit(train_X, train_y)
    # classifier.predict_prob(test_X[0], test_y[0])
    #
    # print("model score: ", classifier.score(test_X, test_y))
