import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from . import config as C


def plot_confusion_matrix(y_true, y_pred, thresh=0.5, class_names=C.TARGET_LABELS, save_name=None):
    cm = multilabel_confusion_matrix(y_true.astype(int), (y_pred>thresh).astype(int))
    cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig,ax = plt.subplots(figsize=(8,6))
    cmd.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    ax.set_title('Normalized Confusion Matrix')
    plt.tight_layout()
    if save_name is not None:
        os.makedirs('imgs/', exist_ok=True)
        #save_path = 'imgs/confusion_matrix.png'
        #if os.path.exists(save_path):
        save_path = 'imgs/'+save_name #inc_fname('confusion_matrix.png','imgs')

        plt.savefig(save_path)
    plt.show()


def plot_cm_ova(labs,preds):
    # multilabel_confusion_matrix(labs5.astype(int),(prds5>0.2).astype(int))
    cms = [confusion_matrix(labs[:,0].astype(int),(preds>0.2)[:,i].astype(int)) for i in range(5)]
    fig,axes = plt.subplots(1,5, figsize=(14,6), sharey=True)
    axes[0].set(ylabel=C.TARGET5_LABELS[0])
    for i,(cm,ax) in enumerate(zip(cms,axes)):
        cmd = ConfusionMatrixDisplay(cm, display_labels=C.TARGET5_LABELS)
        cmd.plot(cmap='Blues', xticks_rotation=45, ax=ax)
        ax.set(title=C.TARGET5_LABELS[i])
        #ax.set(ylabel=C.TARGET5_LABELS[0])
        #ax.set_title('Confusion Matrix')
    plt.tight_layout()


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.

    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = 'd'#'.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in itertools.product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        #fig.colorbar(self.im_, ax=ax)
        #plt.xticks
        ax.set(xticks=[],#np.arange(n_classes),
               yticks=[])#np.arange(n_classes),)
               #xticklabels=self.display_labels,
               #yticklabels=self.display_labels,
               #ylabel="True label",
               #xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self