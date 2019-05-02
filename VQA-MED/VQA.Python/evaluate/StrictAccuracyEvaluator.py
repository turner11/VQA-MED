import re
import pandas as pd

from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase


# noinspection PyMethodMayBeStatic
class StrictAccuracyEvaluator(VqaMedEvaluatorBase):

    def __init__(self, predictions: iter, ground_truth: iter):
        super().__init__(predictions, ground_truth)
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't",
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                             "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've",
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                             "Id've": "I'd've", "I'dve": "I'd've",
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                             "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                             "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't",
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                             "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                             "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                             "someone'dve": "someone'd've",
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                             "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've",
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                             "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't",
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                             "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've",
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                             "whos": "who's", "whove": "who've", "whyll": "why'll",
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                             "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've",
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                             "youd've": "you'd've", "you'dve": "you'd've",
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]
        self.period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
        # self.comma_strip = re.compile("(\d)(\,)(\d)")
        self.comma_strip = re.compile("(\d)(,)(\d)")
        self.punctuation = [';', r"/", '[', ']', '"', '{', '}',
                            '(', ')', '=', '+', '\\', '_', '-',
                            '>', '<', '@', '`', ',', '?', '!']

    def evaluate(self) -> float:
        predictions = self.predictions
        ground_truth = self.ground_truth

        # Compute score
        accuracy = self._compute_accuracy(predictions, ground_truth)
        return accuracy

    def get_name(self):
        return 'strict_accuracy'

    def _compute_accuracy(self, predictions: iter, ground_truth: iter) -> float:
        df = pd.DataFrame({'prediction': [pred.answer for pred in predictions],
                           'ground_truth': [gt.answer for gt in ground_truth]})

        def clean_predictions(p):
            prediction = self._clean_prediction_spaces(p)
            normlized_prediction = self._normalize(prediction)
            return normlized_prediction

        df['normalized_prediction'] = df.prediction.apply(lambda p: clean_predictions(p))
        df['normalized_ground_truth'] = df.ground_truth.apply(self._normalize)
        df['matches'] = df.normalized_prediction == df.normalized_ground_truth

        accuracy = float(sum(df.matches)) / len(df)
        return accuracy

    def _normalize(self, text):
        text = self._process_punctuation(text)
        text = self._process_digit_article(text)
        return text

    def _clean_prediction_spaces(self, prediction: str) -> str:
        prediction = prediction.replace('\n', ' ')
        prediction = prediction.replace('\t', ' ')
        prediction = prediction.strip()
        return prediction

    def _process_punctuation(self, text):
        out_text = text
        for p in self.punctuation:
            if (p + ' ' in text or ' ' + p in text) or (re.search(self.comma_strip, text) is not None):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.period_strip.sub("", out_text, re.UNICODE)
        return out_text

    def _process_digit_article(self, text):
        out_text = []
        temp_text = text.lower().split()
        for word in temp_text:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
            else:
                pass
        for wordId, word in enumerate(out_text):
            if word in self.contractions:
                out_text[wordId] = self.contractions[word]
        out_text = ' '.join(out_text)
        return out_text


def main():
    predictions_path = 'C:\\Users\\Public\\Documents\\Data\\2019\\submissions\\20190421_1436_41_answers_predictions\\predictions.hdf'
    with pd.HDFStore(predictions_path) as store:
        df_predictions = store['validation']

    predictions = df_predictions.prediction.values
    ground_truth = df_predictions.answer.values
    evaluator = StrictAccuracyEvaluator(predictions, ground_truth)
    accuracy = evaluator.evaluate()
    print(accuracy)
    str()


if __name__ == '__main__':
    main()
