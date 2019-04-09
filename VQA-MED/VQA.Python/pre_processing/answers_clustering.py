from pathlib import Path
from typing import Type, Union
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn import decomposition


import matplotlib.pyplot as plt
import logging
from common.os_utils import File
from common.settings import data_access as data_access_api
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from evaluate.WbssEvaluator import WbssEvaluator
from evaluate.BleuEvaluator import BleuEvaluator

logger = logging.getLogger(__name__)


class SentenceClusterMaker(object):
    """"""
    evaluator_ctor: Type[VqaMedEvaluatorBase]

    def __init__(self, sentences, min_count_for_cluster=5, cluster_eps=0.3, df_distances_pca_n_components=500):
        """"""
        super().__init__()
        self.sentences = sentences
        self.min_count_for_cluster = min_count_for_cluster
        self.cluster_eps = cluster_eps
        self.df_distances_pca_n_components = df_distances_pca_n_components
        self.evaluator_ctor = BleuEvaluator  # WbssEvaluator

    def __repr__(self):
        return super().__repr__()

    def cluster(self, df_distances: Union[pd.DataFrame, str, Path] = None, plot=False):

        save_root = Path('D:\\Users\\avitu\\Downloads\\')
        if df_distances is None:
            df_distances = self._get_distances_matrix(self.sentences, self.evaluator_ctor)

            # p = str(save_root / 'distances.h5')
            # logger.info(f'Writing distances to:\n{p}')
            # with pd.HDFStore(p) as store:
            #     store['distances'] = df_distances
        else:
            df_distances = self.get_df_instances(df_distances)

        logger.debug('Getting clusters')
        clusters = self.get_clusters(df_distances, plot)

        p2 = str(save_root / 'classes.pkl')
        File.dump_pickle(clusters, p2)

        return clusters, df_distances

    @staticmethod
    def get_df_instances(df_distances: Union[pd.DataFrame, str, Path]):
        if isinstance(df_distances, (str, Path)):
            df_distances_path = str(df_distances)
            logger.info(f'Loading distances from:\n{df_distances_path}')
            with pd.HDFStore(df_distances_path) as store:
                df_distances = store['distances']

        assert isinstance(df_distances, pd.DataFrame), \
            f'Expected df_distances to be a dataframe, got {type(df_distances).__name__}'

        return df_distances

    def get_clusters(self, df_distances: pd.DataFrame, plot: bool = False) -> [[str]]:
        return self._get_clusters(df_distances, self.min_count_for_cluster,
                                  self.cluster_eps,
                                  self.df_distances_pca_n_components,
                                  plot)

    @staticmethod
    def _get_clusters(df_distances: Union[pd.DataFrame, str],
                      min_count_for_cluster: float,
                      cluster_eps: float,
                      df_distances_pca_n_components: int,
                      plot: bool = False) -> [[str]]:
        df_distances = SentenceClusterMaker.get_df_instances(df_distances)

        if df_distances_pca_n_components > 0:
            df_pca = SentenceClusterMaker.get_pca(df_distances, df_distances_pca_n_components)
            X = df_pca.values
        else:
            X = df_distances.values

        X = StandardScaler().fit_transform(X)

        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=cluster_eps, min_samples=min_count_for_cluster).fit(X)

        clusters = db.labels_

        # Number of clusters in clusters, ignoring noise if present.
        n_clusters_ = SentenceClusterMaker.__get_clusters_count(clusters)
        n_noise_ = list(clusters).count(-1)

        logger.info(f'Estimated number of clusters: {n_clusters_}')
        logger.info(f'Estimated number of noise points: {n_noise_}')
        # #############################################################################
        # Plot result
        # Black removed and is used for noise instead.

        if plot:
            SentenceClusterMaker._plot(X, db)

        return clusters

    @staticmethod
    def get_pca(df_distances: pd.DataFrame,
                n_components: int,
                normalize : bool = True) -> pd.DataFrame:
        pca = decomposition.PCA(n_components=n_components)
        if normalize:
            x_std = StandardScaler().fit_transform(df_distances)
            pca_input = x_std
        else:
            pca_input = df_distances

        pca_dist = pca.fit_transform(pca_input)
        df_pca = pd.DataFrame(pca_dist)
        return df_pca

    @staticmethod
    def _plot(X: np.ndarray, db: DBSCAN):
        fig, ax = plt.subplots()

        clusters = db.labels_
        n_clusters_ = SentenceClusterMaker.__get_clusters_count(clusters)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        unique_labels = set(clusters)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        colors = [c[:3] for c in colors]
        unique_labels = sorted(unique_labels)  # put -1 first

        dists = []
        cluster_sizes = []
        for label, curr_color in zip(unique_labels, colors):
            points_opacity = 0.7
            if label == -1:
                points_opacity = 0.1
                # Black used for noise.
                curr_color = [0, 0, 0, points_opacity]

            class_member_mask = (clusters == label)

            xy = X[class_member_mask & core_samples_mask]
            cluster_size = len(xy)

            D = pdist(xy)
            Da = squareform(D)
            max_dist = Da.max()
            if max_dist > 0:
                min_dist = Da[Da != 0].min()
            else:
                min_dist = 0

            dists.append(max_dist)
            cluster_sizes.append(cluster_size)
            indices = [idx for idx, cluster in enumerate(clusters) if cluster == label]

            if len(xy) > 0:
                centroid = xy.mean(axis=0)
                centroid_x = centroid[0]
                centroid_y = centroid[1]

                # plot the centroid
                size = min(cluster_size * 3.5, 30)
                line, = plt.plot([centroid_x], [centroid_y], 'o', markerfacecolor=tuple(curr_color),
                                 # markeredgecolor='label',
                                 markersize=size)
                picker = max(5, size * 0.6)
                line.set_picker(picker)  # for hovering not directly above center

                def closest_node(node, nodes):
                    if len(nodes) == 0:
                        return -1
                    nodes = np.asarray(nodes)
                    dist_2 = np.sum((nodes - node) ** 2, axis=1)
                    closest_idx = np.argmin(dist_2)
                    return closest_idx

                closest_node_idx = closest_node(centroid, xy)
                closest = indices[closest_node_idx]

                centroid_tool_tip = f'Cluster "{label}". {cluster_size} sentences\n' \
                    f'std: {np.std(xy)}\n' \
                    f'max dist: {max_dist: .6f}\n' \
                    f'min dist: {min_dist: .6f}\n' \
                    f'indices = [{",".join(str(idx) for idx in indices)}]\n' \
                    f'closest_node = {closest}'
                tool_tips = [centroid_tool_tip]
                SentenceClusterMaker.__connect_events(fig, line, ax, tool_tips)

            ## This will plot all points
            points_line, = plt.plot(xy[:, 0], xy[:, 1], 'x', color=curr_color,
                                    # markeredgecolor='label',
                                    markersize=6,
                                    alpha=points_opacity)

            points_tool_tips = [f'cluster {label}'] * len(xy)
            SentenceClusterMaker.__connect_events(fig, points_line, ax, points_tool_tips)

            xy_none_core = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy_none_core[:, 0], xy_none_core[:, 1], 'o', markerfacecolor=tuple(curr_color),
                     # markeredgecolor='label',
                     markersize=6)

        print(f'largest dist: {max(dists) :.6f} ')
        print(f'Cluster sizes: {Counter(cluster_sizes)}')
        reduction = sum(
            [cluster_count * cluster_elements for cluster_count, cluster_elements in Counter(cluster_sizes).items()]) \
                    - len(unique_labels)
        print(f'This will reduce number of answers by: {reduction}')
        plt.title(f'Estimated number of clusters: {n_clusters_}. Reduction: {reduction}')
        plt.show()


    @staticmethod
    def __connect_events(fig, line, ax, tool_tips):

        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))

        def update_annot(ind, tt=tool_tips, line_arg=line):
            x, y = line_arg.get_data()
            annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
            text_set = set([tt[n] for n in ind["ind"]])
            text = '\n'.join(text_set)

            # text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
            #                        " ".join([tt[n] for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event, ax_arg=ax, line_arg=line, fig_arg=fig):
            vis = annot.get_visible()
            if event.inaxes == ax_arg:
                cont, ind = line_arg.contains(event)
                if cont:
                    # print(f'hover: cont - ind: {ind}')
                    update_annot(ind)
                    annot.set_visible(True)
                    fig_arg.canvas.draw_idle()
                else:
                    if vis:
                        # print(f'hover: NOT cont - ind: {ind}')
                        # print('hover: vis - turning off')
                        annot.set_visible(False)
                        fig_arg.canvas.draw_idle()

        def onclick(event, line_arg=line):
            is_vis = annot.get_visible()
            if not is_vis:
                # print('Not visible')
                return
            if not event.dblclick:
                if event.button == 1:
                    cont, ind = line_arg.contains(event)
                    if cont:
                        update_annot(ind)

                    txt = annot.get_text()
                    print(txt)

                elif event.button == 3:
                    # Write to figure
                    pass
                else:
                    pass  # Do nothing

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect('button_press_event', onclick)

    @staticmethod
    def __get_clusters_count(raw_clusters):
        n_clusters_ = len(set(raw_clusters)) - (1 if -1 in raw_clusters else 0)
        return n_clusters_

    @staticmethod
    def _get_distances_matrix(sentences: [str], evaluator_ctor: Type[VqaMedEvaluatorBase]) -> pd.DataFrame:
        sentence_idxs = list(range(len(sentences)))
        df_distances = pd.DataFrame(index=sentence_idxs, columns=sentence_idxs, dtype=float)
        pbar_row = tqdm(sentence_idxs)
        for row_idx in pbar_row:
            answer = sentences[row_idx]
            # pbar_col = tqdm(sentence_idxs)
            for col_idx in sentence_idxs:
                current_score = df_distances.loc[row_idx, col_idx]
                if not np.isnan(current_score):
                    continue

                ground_truth_reference = sentences[col_idx]
                evaluator = evaluator_ctor(predictions=[answer], ground_truth=[ground_truth_reference])
                score = evaluator.evaluate()
                df_distances.loc[row_idx, col_idx] = score
                df_distances.loc[col_idx, row_idx] = score

        return df_distances


def clustering_main():
    meta = data_access_api.load_meta()

    df_answers = meta['answers']
    df_answers = df_answers[df_answers.question_category == 'Abnormality']
    sentences = df_answers.processed_answer.values

    # sentences = sentences[:50]
    cluster_maker = SentenceClusterMaker(sentences)
    clusters, distances = cluster_maker.cluster(plot=True)

    str()


if __name__ == '__main__':
    clustering_main()
