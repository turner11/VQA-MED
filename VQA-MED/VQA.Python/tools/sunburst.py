from collections import Counter
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from common.settings import data_access



def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center')

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()


    str()


def main():
    df = data_access.load_processed_data()
    df = df[df.question_category == 'Abnormality']
    sentences = df.processed_question


    for i in range(10):
        data = get_data(sentences, 20)
        sunburst(data)
        plt.show(block=True)
    str()


def get_data(sentences, sample_count=None):
    if sample_count:
        sentences = sentences.sample(20)

    stops = set(stopwords.words("english"))# - {'what', 'was','where'}

    def remove_stops(sentence):
        words = [word for word in sentence if word.lower() not in stops]
        return words


    words_lists = sentences.apply(str.lower).apply(str.split)#.apply(remove_stops)

    def get_relevant_sentences(prefix_list):
        idxs = words_lists.apply(lambda lst: lst[:len(prefix_list)] == prefix_list)
        return words_lists[idxs]

    def get_layer_words(prefix_list):
        layer = len(prefix_list)
        relevant_sentence_words = get_relevant_sentences(prefix_list)
        lsdt_gen = (lst for lst in relevant_sentence_words if len(lst) > layer)
        return [lst[layer] for lst in lsdt_gen]

    def get_data_recursive(prefix):
        relevant_words = get_layer_words(prefix)
        counter = Counter(relevant_words)
        sub_datas = []
        for word, count in counter.items():
            curr_prefix = prefix+[word]
            word_sub_data = get_data_recursive(curr_prefix)
            curr_word_data = [word, count, word_sub_data]
            sub_datas.append(curr_word_data)

        return sub_datas


    data = get_data_recursive(prefix=[])
    return data

    data = [
        ('/', 100, [
            ('home', 70, [
                ('Images', 40, []),
                ('Videos', 20, []),
                ('Documents', 5, []),
            ]),
            ('usr', 15, [
                ('src', 6, [
                    ('linux-headers', 4, []),
                    ('virtualbox', 1, []),

                ]),
                ('lib', 4, []),
                ('share', 2, []),
                ('bin', 1, []),
                ('local', 1, []),
                ('include', 1, []),
            ]),
        ]),
    ]
    return data


if __name__ == '__main__':
    main()


