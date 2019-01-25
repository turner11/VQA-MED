from common.DAL import get_models, get_scores


def test_getting_models():
    models = get_models()
    assert len(models) > 0


def test_getting_scores():
    scores = get_scores()
    assert len(scores) > 0
