from argparse import ArgumentError
from collections import namedtuple
from typing import List, Union
import nltk

Prediction = namedtuple('Prediction', ['q_id', 'image_id', 'answer'])


class VqaMedEvaluatorBase(object):
    """
    Evaluator class
    Evaluates one single runfile
    _evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
    """

    def __init__(self, predictions: iter, ground_truth: iter):
        """
        Constructor
        Parameter 'answer_file_path': Path of file containing ground truth
        """
        predictions, ground_truth = self.consolidate_input(predictions, ground_truth)
        self.predictions = predictions
        # Ground truth data
        self.ground_truth = ground_truth

    def get_name(self):
        raise NotImplementedError("Evaluation should be implemented by a concrete class")

    def evaluate(self):
        """
        This is the only method that will be called by the framework
        Parameter 'submission_file_path': Path of the submitted runfile
        returns a _result_object that can contain up to 2 different scores
        """
        raise NotImplementedError("Evaluation should be implemented by a concrete class")

    @staticmethod
    def update_nltk():
        # Note: This could also be breaken down to classes.
        # I'm not doing it since I want everything to be updated with a single call,
        # especially because it does not take that much of time
        nltk.download('wordnet')
        # NLTK
        # Download Punkt tokenizer (for word_tokenize method)
        # Download stopwords (for stopword removal)
        nltk.download('punkt')
        nltk.download('stopwords')

    @classmethod
    def get_all_evaluation(cls, predictions: List[tuple], ground_truth: List[tuple]):
        # cls.update_nltk()
        from evaluate.BleuEvaluator import BleuEvaluator
        from evaluate.WbssEvaluator import WbssEvaluator
        sub_classes = [BleuEvaluator, WbssEvaluator]
        instances = [sub_cls(predictions, ground_truth) for sub_cls in sub_classes]

        evaluations = {ins.get_name(): ins.evaluate() for ins in instances}

        return evaluations

    @staticmethod
    def consolidate_input(predictions: iter, ground_truth: iter) -> (List[Prediction], List[Prediction]):
        all_types = {type(p) for p in set(predictions).union(set(ground_truth))}
        assert len(all_types) == 1, \
            'Expected all predictions & ground truth to have same type'

        assert len(predictions) == len(ground_truth), \
            'expected predictions and ground truth to be of same length'

        t = all_types.pop()
        if t == str:
            def str2tpl(arr):
                return [Prediction(q_id=i, image_id=i, answer=answer) for i, answer in enumerate(arr)]

            predictions = str2tpl(predictions)
            ground_truth = str2tpl(ground_truth)
        elif t == tuple:
            pass
        else:
            raise Exception(f'Can not handle arguments of type {t}')

        return predictions, ground_truth


def main():
    """
    Test evaluation a set of predictions
    call _evaluate method
    """
    # Create instance of Evaluator
    # predictions = [(0, 0, 'ct')]
    # ground_truth = [(0, 0, 'axial CT')]

    predictions = ['ct']
    ground_truth = ['axial CT']
    result = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    # evaluator = VqaMedEvaluator(predictions=predictions, ground_truth=ground_truth)
    ## Call _evaluate method
    # result = evaluator._evaluate()
    print(result)


if __name__ == "__main__":
    main()
