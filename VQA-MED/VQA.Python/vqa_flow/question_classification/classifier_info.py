import numpy as np
import matplotlib.pyplot as plt


class ClassifierInfo(object):

    def __init__(self, classifier, scores_train, scores_test, classes) -> None:
        super().__init__()
        self.classifier = classifier
        self.scores_train = scores_train
        self.scores_test = scores_test
        self.classes = classes

    @property
    def accuracy(self):
        return self.__get_test_score('accuracy')

    @property
    def f1(self):
        return self.__get_test_score('f1')

    @property
    def precision(self):
        return self.__get_test_score('precision')

    @property
    def recall(self):
        return self.__get_test_score('recall')

    def __get_test_score(self, metric):
        return self.__get_score(metric, self.scores_test)

    def __get_train_score(self, metric):
        return self.__get_score(metric, self.scores_train)

    def __get_score(self, metric, metric_container):
        return metric_container[metric][-1]

    def get_figure(self, title=None, plot=False):
        title = title if title is not None else f'Hidden layers: {self.classifier.hidden_layer_sizes}'
        return self.get_classifier_scores_figure(self.scores_test, self.scores_train, title=title, plot=plot)

    @staticmethod
    def get_classifier_scores_figure(test_vals, train_vals, title='', plot=False):
        fig = None
        try:

            major_ticks = np.arange(0, 1.1, 0.1)
            minor_ticks = np.arange(0, 1.1, 0.05)

            # plt.gcf().clear()
            idx = -1
            plot_items = list(train_vals.keys())
            fig, ax = plt.subplots(nrows=2, ncols=2, sharey='row')  # ,sharex='col'
            for row in ax:
                for col in row:
                    idx += 1
                    name = plot_items[idx]

                    scores_train = train_vals[name]
                    scores_test = test_vals[name]

                    col.plot(scores_train, color='green', alpha=0.8, label='Train')
                    col.plot(scores_test, color='blue', alpha=0.8, label='Test')

                    col.set_yticks(major_ticks)
                    col.set_yticks(minor_ticks, minor=True)

                    col.set(xlabel='Epochs', ylabel=name, title=f"{name} over epochs {scores_test[-1]:.3f}", ylim=[0, 1])  # set_xlim =[0, 5]
            fig.legend(loc='upper left')
            fig.suptitle(str(title), fontsize=20)


        except Exception as ex:
            print(ex)

        if plot:
            plt.pause(0.001)
            plt.show()
        return fig
