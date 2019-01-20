import pytest

from evaluate.BleuEvaluator import BleuEvaluator
from evaluate.WbssEvaluator import WbssEvaluator


@pytest.mark.parametrize("word, similar_word, non_similar_word",
                         [
                            ('ct', 'mri', 'head'),
                            ('stomach', 'abdomen', 'tumor'),
                            ('abdomen', 'stomach', 'chest'),
                            ('tumor', 'cancer', 'fracture'),
                            ('Magnetic resonance imaging', 'mri', 'ct'),
                         ])
def test_bss_relative(word, similar_word, non_similar_word):
    evaluator_ctor = WbssEvaluator
    _test_evaluator_relative(evaluator_ctor ,word, similar_word, non_similar_word)


@pytest.mark.parametrize("word, similar_word, non_similar_word",
                         [
                            ('The roof is on fire', 'roof fire', 'you are fired'),
                             ('group of words to subtract ngrams long or short',
                              'words subtract ngrams long',
                               'words subtract ngrams'),

                         ])
def test_bleu_relative(word, similar_word, non_similar_word):
    evaluator_ctor = BleuEvaluator
    _test_evaluator_relative(evaluator_ctor ,word, similar_word, non_similar_word)

def _test_evaluator_relative(evaluator_ctor ,word, similar_word, non_similar_word):


    evaluations = []
    for predicted_word in [similar_word, non_similar_word]:
        predictions = [predicted_word ]
        ground_truths = [word]
        evaluator = evaluator_ctor(predictions, ground_truth=ground_truths)

        evaluation = evaluator.evaluate()
        evaluations.append(evaluation)


    similar_evaluation = evaluations[0]
    non_similar_evaluation = evaluations[1]

    message = f'Expected "{similar_word}" to be graded better than  "{non_similar_word}" when comparing to {word} ' \
        f'but got an evaluation of  {similar_evaluation } (<={non_similar_evaluation })'
    assert similar_evaluation > non_similar_evaluation, message




@pytest.mark.parametrize("prediction, ground_truth, expected_evaluation, mode",
                         [
                             ('stomach', 'abdomen', 1, 'exact'),
                             ('stomach', 'cancer', 0.1, 'max'),
                             ('foot', 'foot', 1, 'exact'),
                             ('dog food', 'cat meal', 0.2, 'max'),
                             ('scan', 'image', 0.9, 'min'),
                             ('thorax', 'chest', 1, 'exact'),
                             ('chest', 'thorax', 1, 'exact'),
                             ('ct', 'CT', 1, 'exact'),
                         ]
                         )
def test_bss_absolute(prediction, ground_truth, expected_evaluation, mode):
    evaluator_ctor = WbssEvaluator
    _test_evaluator_absolute(evaluator_ctor, prediction, ground_truth, expected_evaluation, mode)


long_sentence: str = 'BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which '\
                     'has been machine-translated from one natural language to another. Quality is considered to be ' \
                     'the correspondence between a machines output and that of a human: "the closer a machine ' \
                     'translation is to a professional human translation, the better it is" – this is the central ' \
                     'idea behind BLEU.[1][2] BLEU was one of the first metrics to claim a high correlation with ' \
                     'human judgements of quality,[3][4] and remains one of the most popular automated and ' \
                     'inexpensive metrics. '


@pytest.mark.parametrize("prediction, ground_truth, expected_evaluation, mode",
                         [
                             ('ct', 'CT', 1, 'exact'),
                             ('bilateral multiple pulmonary nodules', 'bilateral MULTIPLE pulmonary NODULES', 1,
                              'exact'),
                             ('no sub ngrams', 'a sentence with words', 0, 'exact'),
                             ('lets say something', 'say something', 0.5, 'min'),
                             (long_sentence + ' a short suffix', long_sentence, 0.95, 'min'),
                             (long_sentence, long_sentence + ' a short suffix', 0.95, 'min'),

                         ])
def test_bleu_absolute(prediction, ground_truth, expected_evaluation, mode):
    evaluator_ctor = BleuEvaluator
    _test_evaluator_absolute(evaluator_ctor, prediction, ground_truth, expected_evaluation, mode)


def _test_evaluator_absolute(evaluator_ctor, prediction, ground_truth, expected_evaluation, mode):
    predictions = [prediction]
    ground_truths = [ground_truth]
    evaluator = evaluator_ctor(predictions, ground_truth=ground_truths)
    actual_evaluation = evaluator.evaluate()

    rounded_valuation = round(actual_evaluation, ndigits=6)
    message = f'for prediction "{prediction}" and ground truth of "{ground_truth}" ' \
        f'expected to get an evaluations of {mode} {expected_evaluation} but got {rounded_valuation}'
    if mode == 'exact':
        assert pytest.approx(actual_evaluation, 0.01) == expected_evaluation, message
    elif mode == 'max':
        assert actual_evaluation <= expected_evaluation, message
    elif mode == 'min':
        assert actual_evaluation >= expected_evaluation, message

    str()


if __name__ == '__main__':
    test_bleu_absolute('ct', 'CT', 1, 'exact')
    test_bleu_absolute('ct ct ct ct ct ct ct ct ct ct ct ct', 'CT', 1, 'exact')

    test_bss_absolute('ct', 'mri', 2, 'exact')

    test_bss_absolute('dog food', 'cat meal', 1, 'exact')
    test_bss_absolute('foot', 'foot', 1, 'exact')
    test_bss_absolute('stomach', 'cancer', 0.1, 'max')
    test_bss_absolute('stomach', 'abdomen', 1, 'exact')
