import pytest
from itertools import combinations

from pre_processing.answers_clustering import SentenceClusterMaker


def _get_n_same_sentences(n_sentences, duplication_factor=5):
    words = combinations('abcdefghijklmnopqrstuvwkyz', r=4)
    ret = []
    for i in range(n_sentences):
        word_letters = next(words)
        word = ''.join(word_letters)
        sentence = ' '.join([word for i in range(5)])
        curr_bulk = [sentence for j in range(duplication_factor)]
        ret.extend(curr_bulk)
    return ret


@pytest.fixture(scope='session')
def clusterd_duplicate_sentence():
    sentences = ['same sentence repeat itself'] * 10
    cluster_maker = SentenceClusterMaker(sentences,
                                         min_count_for_cluster=2,
                                         df_distances_pca_n_components=0)
    clusters, distances = cluster_maker.cluster()
    return sentences, clusters, distances


def test_single_cluster(clusterd_duplicate_sentence):
    sentences, clusters, distances = clusterd_duplicate_sentence
    assert len(set(clusters)) == 1, 'Expected to get a single clusters'


@pytest.mark.parametrize('n_sentences', range(1,6))
def test_number_of_cluster(n_sentences):
    sentences = _get_n_same_sentences(n_sentences)

    min_cluster = n_sentences#min(5, n_sentences)
    cluster_maker = SentenceClusterMaker(sentences,
                                         min_count_for_cluster=min_cluster,
                                         df_distances_pca_n_components=0)
    clusters, distances = cluster_maker.cluster()
    uniques_clusters = set(clusters)

    assert len(uniques_clusters) == n_sentences


# @pytest.mark.skip(reason='WIP')
def test_correct_dimensions(clusterd_duplicate_sentence):
    sentences, clusters, distances = clusterd_duplicate_sentence
    assert len(distances) == len(sentences)
    assert len(distances.columns) == len(sentences)


def main():
    test_number_of_cluster(n_sentences=5)
    tpl = clusterd_duplicate_sentence()
    test_single_cluster(tpl)
    test_correct_dimensions(tpl)


if __name__ == '__main__':
    main()
