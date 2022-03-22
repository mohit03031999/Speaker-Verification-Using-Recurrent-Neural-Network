## This method is copied from plot_confusion in confusion.py without the
'''
@author: Roch
'''

import matplotlib.pyplot as plt
import numpy as np
import itertools


## section that calculates the confusion matrix
def plot_confusion(cumulative, N, labels):
    # Build the figure and axes
    fig = plt.figure(figsize=(1.5,1.5), dpi=200,
                    facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)


    # Plot the confusion matrix
    # Show the heat map relative to the number of true examples
    # so that the most frequent decisions are always highlighted
    relative = cumulative / cumulative.sum(axis=1)[:,None]
    im = ax.imshow(relative, cmap='Oranges')

    # Label the axes
    tick_marks = np.arange(N)

    ax.set_xlabel('Predicted', fontsize=5)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=3.5,  ha='center')
    ax.xaxis.set_label_position('bottom')

    ax.set_ylabel('True Label', fontsize=5)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    # Add counts in boxes
    for i, j in itertools.product(
        range(cumulative.shape[0]), range(cumulative.shape[1])):

        ax.text(j, i, format(cumulative[i, j], 'd') if cumulative[i,j] !=0
                else '.',
                horizontalalignment="center", fontsize=2,
                verticalalignment='center', color= "black")

    fig.set_tight_layout(True)
    fig.show()

    return fig, ax, im

lst = [[ 2, 0, 0, 8], [ 0,  2,  0,  8], [ 0,  0,  6,  4], [ 0,  0, 17, 33]]
plot_confusion(np.array(lst), 4, ['Jacob', 'Mohit', 'Vivek', 'Others'])
