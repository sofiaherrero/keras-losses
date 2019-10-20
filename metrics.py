from keras import backend as K


def recall(y_true, y_pred):
    """
    Calculates recall as: tp / tp + fn, where tp + fn is also the total number of positives.

    Args:
        y_true: 1D tensor, true targets
        y_pred: 1D tensor, predicted targets

    """

    n_true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    n_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = n_true_positives / (n_positives + K.epsilon())

    return recall


def precision(y_true, y_pred):
    """
       Calculates precision as: tp / tp + fp, where tp + fp is also the total number of predicted positives.

       Args:
           y_true: 1D tensor, true targets
           y_pred: 1D tensor, predicted targets

    """

    n_true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    n_predictive_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = n_true_positives / (n_predictive_positives + K.epsilon())

    return precision


def f1_score(y_true, y_pred):
    """
       Calculates F1 score as: 2 * (precision * recall / precision + recall)

       Args:
           y_true: 1D tensor, true targets
           y_pred: 1D tensor, predicted targets

    """
    precision_result = precision(y_true, y_pred)
    recall_result = recall(y_true, y_pred)
    return 2 * ((precision_result * recall_result) / (precision_result + recall_result + K.epsilon()))
