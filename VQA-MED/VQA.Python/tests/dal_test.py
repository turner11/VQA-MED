import pytest
from common.DAL import get_models, get_scores

@pytest.mark.skip(reason='No models for 2019 yet')
def test_getting_models():
    models = get_models()
    assert len(models) > 0


@pytest.mark.skip(reason='No models for 2019 yet')
def test_getting_scores():
    scores = get_scores()
    assert len(scores) > 0
