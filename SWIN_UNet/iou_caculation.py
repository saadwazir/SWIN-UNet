import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

## IOU in pure numpy
def numpy_iou(y_true, y_pred, n_class=3):
    def iou(y_true, y_pred, n_class):
        # IOU = TP/(TP+FN+FP)
        IOU = []
        for c in range(n_class):
            TP = np.sum((y_true == c) & (y_pred == c))
            FP = np.sum((y_true != c) & (y_pred == c))
            FN = np.sum((y_true == c) & (y_pred != c))

            n = TP
            d = float(TP + FP + FN + 1e-12)

            iou = np.divide(n, d)
            IOU.append(iou)

        return np.mean(IOU)

    batch = y_true.shape[0]
    y_true = np.reshape(y_true, (batch, -1))
    y_pred = np.reshape(y_pred, (batch, -1))

    score = []
    for idx in range(batch):
        iou_value = iou(y_true[idx], y_pred[idx], n_class)
        score.append(iou_value)
    return np.mean(score)


## Calculating IOU across a range of thresholds, then we will mean all the
## values of IOU's.
## this function can be used as keras metrics
def numpy_mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 1.0):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# def tf_mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.5):
#         y_pred_ = tf.cast(y_pred > t, tf.int32)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         prec.append(score)
#     val = K.mean(K.stack(prec), axis=0)
#     return [val, up_opt]


# r = numpy_mean_iou(test_target, y_pred)
# print(r)