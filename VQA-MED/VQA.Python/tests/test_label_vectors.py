import pandas as pd
import pytest

from common.functions import sentences_to_hot_vector, hot_vector_to_words

df_answers = pd.DataFrame({'answer':
                               ['how are you this morning?'
                                   , 'I am fine, And how are you?'
                                   , 'good']})

words = ['good', 'how', 'fine']


@pytest.mark.parametrize("labels, expected_hot_vectors",
                         [
                             (['good'], [[1, 0, 0]]),
                             (['how'], [[0, 1, 0]]),
                             (['fine'], [[0, 0, 1]]),
                             (['good how'], [[1, 1, 0]]),
                             (['good fine'], [[1, 0, 1]]),
                             (['how fine'], [[0, 1, 1]]),
                             (['good how fine'], [[1, 1, 1]]),
                             (['fine stranger'], [[0, 0, 1]]),
                             (['no words from vocab'], [[0, 0, 0]]),
                             (['fine_with_suffix'], [[0, 0, 0]]),

                             (['good', 'how'], [[1, 0, 0], [0, 1, 0]]),

                         ])
def test_word_labeling(labels, expected_hot_vectors):
    # Arrange
    classes = words
    # Act - Create the hot vector
    arr_one_hot_vector = sentences_to_hot_vector(labels, classes)
    assert len(arr_one_hot_vector) == len(labels)

    for expected_hot_vector, hot_vector in zip(expected_hot_vectors, arr_one_hot_vector):
        assert list(expected_hot_vector) == list(hot_vector)


def test_sentence_labeling():
    # Arrange
    classes = df_answers.answer
    labels = classes

    # Act - Create the hot vector
    arr_hot_vector = sentences_to_hot_vector(labels, classes)

    for idx, lbl in enumerate(labels.values):
        one_hot_vector = arr_hot_vector[idx]
        df_sentence_from_vector = hot_vector_to_words(one_hot_vector, labels)
        assert len(df_sentence_from_vector) == 1, f'Expected to get a single results for a single vector ' \
            f'but got {len(df_sentence_from_vector)}'
        label_words = df_sentence_from_vector.values[0]

        # Assert: Make sure the reverse gives you the correct label
        assert label_words == lbl, f'Expected to get label "{lbl}", but got {label_words}'


def main():
    test_sentence_labeling()
    test_word_labeling(['good fine'], [[1, 0, 1]])
    test_word_labeling(['good'],[[1, 0, 0]])
    str()


if __name__ == '__main__':
    main()
