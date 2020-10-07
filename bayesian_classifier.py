class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        self._Poe = {'_number_': 0}
        self._Lovecraft = {'_number_': 0}
        self._Shelley = {'_number_': 0}
        self._all = {'_number_': 0}
        self._poe_name = 'Edgar Alan Poe'
        self._shelley_name = 'Mary Wollstonecraft Shelley '
        self._lovecraft_name = 'HP Lovecraft'

    def fit(self, dataframe):
        """
        Fit Naive Bayes parameters according to train data message and authors.
        :param message: pd.DataFrame|list - train input/messages
        :param authors: pd.DataFrame|list - train output/labels
        :return: None
        """
        for index, text in dataframe.iterrows():
            message = text['text']
            author = text['author']
            author_dict = {}

            if author == self._shelley_name:
                author_dict = self._Shelley
            elif author == self._poe_name:
                author_dict = self._Poe
            elif author == self._lovecraft_name:
                author_dict = self._Lovecraft

            for word in message:
                self._all['_number_'] += 1
                author_dict['_number_'] += 1
                if word not in self._all:
                    self._all[word] = 1
                else:
                    self._all[word] += 1

                if word not in author_dict:
                    author_dict[word] = 1
                else:
                    author_dict[word] += 1

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        probability = 1
        author = ""

        if label == self._shelley_name:
            author = self._Shelley
        elif label == self._poe_name:
            author = self._Poe
        elif label == self._lovecraft_name:
            author = self._Lovecraft

        for word in message:
            if word in author:
                probability *= (author[word]/author['_number_'])/\
                               (self._all[word]/self._all['_number_'])

        return probability*(1/3)

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        poe_prob = self.predict_prob(message, self._poe_name)
        lovecraft_prob = self.predict_prob(message, self._lovecraft_name)
        shelley_prob = self.predict_prob(message, self._shelley_name)

        predict_prob = max(poe_prob, lovecraft_prob, shelley_prob)
        if poe_prob == predict_prob:
            return self._poe_name
        elif lovecraft_prob == predict_prob:
            return self._lovecraft_name
        else:
            return self._shelley_name

    def score(self, dataframe):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        score = 0
        number = 0
        for index, text in dataframe.iterrows():
            prediction = self.predict(text['text'])
            if prediction == text['author']:
                score += 1
            number += 1

        return round((score/number)*100, 2)