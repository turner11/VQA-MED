import functools
from keras import backend as K

@functools.lru_cache(maxsize=32)
def _get_base_values(y_pred, y_true):
    def sum_round_clip(val):
        return K.sum(K.round(K.clip(val, 0, 1)))

    # Count positive samples.
    c1 = sum_round_clip(y_true * y_pred)
    c2 = sum_round_clip(y_pred)
    c3 = sum_round_clip(y_true)
    return c1, c2, c3


def precision_score(y_pred, y_true):
    c1, c2, c3 = _get_base_values(y_pred, y_true)
    # How many selected items are relevant?
    precision = c1 / c2
    return precision

def recall_score(y_pred, y_true):
    c1, c2, c3 = _get_base_values(y_pred, y_true)
    # How many selected items are relevant?

    # If there are no true samples, fix the recall score at 0.
    if c3 == 0:
        return 0
    recall = c1 / c3
    return recall


def f1_score(y_true, y_pred):
    recall = recall_score(y_pred, y_true)
    # If there are no relevant samples selected, fix the F1 score at 0.
    if recall == 0:
        return 0

    precision = precision_score(y_pred, y_true)

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    if f1_score is None:
        f1_score = 0
    return f1_score
