from typing import Type
import pandas as pd
from tqdm import tqdm

from common.os_utils import File
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from evaluate.WbssEvaluator import WbssEvaluator
from evaluate.BleuEvaluator import BleuEvaluator

from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

class SenteceClusterMaker(object):
    """"""
    evaluator_ctor: Type[VqaMedEvaluatorBase]

    def __init__(self, sentences):
        """"""
        super().__init__()
        self.sentences = sentences
        self.evaluator_ctor = WbssEvaluator# WbssEvaluator

    def __repr__(self):
        return super().__repr__()

    def cluster(self):
        df_distances = self._get_distances_matrix(self.sentences, self.evaluator_ctor)
        
        root ='C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\'
        p = root +'distances.h5'
        with pd.DataFrame(p) as store:
            store['distances'] = p

        clusters = self._get_clusters(df_distances)

        p2 = root+'classes.pkl'
        File.dump_pickle(clusters,p2)

        return clusters, df_distances

    def _get_clusters(self, df_distances: pd.DataFrame) -> [[str]]:
        # labels_true = [1]*len(df_distances)
        X = df_distances.values
        X = StandardScaler().fit_transform(X)

        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        clusters = db.labels_

        # Number of clusters in clusters, ignoring noise if present.
        n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise_ = list(clusters).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        # #############################################################################
        # Plot result
        # Black removed and is used for noise instead.
        plot_results = True
        if plot_results:
            unique_labels = set(clusters)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (clusters == k)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()

        return clusters

    @staticmethod
    def _get_distances_matrix(sentences: [str], evaluator_ctor: Type[VqaMedEvaluatorBase]) -> pd.DataFrame:
        sentence_idxs = list(range(len(sentences)))
        df_distances = pd.DataFrame(index=sentence_idxs, columns=sentence_idxs, dtype=float)
        pbar_row = tqdm(sentence_idxs)
        for row_idx in pbar_row:
            answer = sentences[row_idx]
            # pbar_col = tqdm(sentence_idxs)
            for col_idx in sentence_idxs:
                current_score = df_distances.loc[row_idx,col_idx]
                if not np.isnan(current_score):
                    continue

                ground_truth_reference = sentences[col_idx]
                evaluator = evaluator_ctor(predictions=[answer], ground_truth=[ground_truth_reference])
                score = evaluator.evaluate()
                df_distances.loc[row_idx,col_idx] = score
                df_distances.loc[col_idx,row_idx] = score

        return df_distances

def main():
    from tests import root
    meta_location = root/'data_for_test\\test_model\\meta_data.h5'
    with pd.HDFStore(str(meta_location)) as store:
        df_answers = store['answers']
        sentences = df_answers.answer.values

        # sentences = sentences[:50]
        cluster_maker = SenteceClusterMaker(sentences)
        clusters, distances = cluster_maker.cluster()

        str()


if __name__ == '__main__':
    main()
