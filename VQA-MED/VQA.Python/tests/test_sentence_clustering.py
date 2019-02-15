import pytest

from pre_processing.answers_clustering import SenteceClusterMaker

@pytest.fixture()
def clusterd_duplicate_sentence():
    sentences = ['same sentence repeat itself'] * 10
    cluster_maker = SenteceClusterMaker(sentences)
    clusters, distances = cluster_maker.cluster()
    return sentences ,clusters, distances

@pytest.mark.skip(reason='WIP')
def test_number_of_cluster(clusterd_duplicate_sentence):
    sentences, clusters, distances = clusterd_duplicate_sentence
    assert len(clusters) == 1

@pytest.mark.skip(reason='WIP')
def test_correct_dimensions(clusterd_duplicate_sentence):
    sentences, clusters, distances = clusterd_duplicate_sentence
    assert len(distances) == len(sentences)
    assert len(distances.columns) == len(sentences)

def main():
    tpl = clusterd_duplicate_sentence()
    test_number_of_cluster(tpl)
    test_correct_dimensions(tpl)

if __name__ == '__main__':
    main()
