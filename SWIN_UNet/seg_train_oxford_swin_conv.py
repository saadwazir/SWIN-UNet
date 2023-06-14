g_n = 5

import os
from glob import glob

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import logging
import random
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Input, concatenate
from tensorflow.keras.models import Model

from iou_caculation import numpy_mean_iou
from keras_vision_transformer import swin_layers, transformer_layers, utils
from swin_unet import swin_unet_2d_conv

logging.getLogger("tensorflow").setLevel(logging.FATAL)


RAND_SEED = 42
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)

"""
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
"""


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[g_n], "GPU")
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    import os

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    # filepath = "./dataset/oxford_iiit/"
    filepath = "/mnt/hdd_2A/segment_ai_ml_project/datasets/oxford_pets_clean/"

    filter_num_begin = 128  # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
    stack_num_down = 2  # number of Swin Transformers per downsampling level
    stack_num_up = 2  # number of Swin Transformers per upsampling level
    patch_size = (
        4,
        4,
    )  # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    window_size = [12, 12, 12, 12]
    num_mlp = 512  # number of MLP nodes within the Transformer
    shift_window = True  # Apply window shifting, i.e., Swin-MSA

    # filter_num_begin = 128  # number of channels in the first downsampling block; it is also the number of embedded dimensions
    # depth = 4  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
    # stack_num_down = 2  # number of Swin Transformers per downsampling level
    # stack_num_up = 2  # number of Swin Transformers per upsampling level
    # patch_size = (
    #     4,
    #     4,
    # )  # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    # num_heads = [4, 8, 8, 8]  # number of attention heads per down/upsampling level
    # window_size = [4, 2, 2, 2]  # the size of attention window per down/upsampling level
    # num_mlp = 512  # number of MLP nodes within the Transformer
    # shift_window = True  # Apply window shifting, i.e., Swin-MSA

    # Input section
    input_size = (384, 384, 3)
    IN = Input(input_size)

    # Base architecture
    X = swin_unet_2d_conv(
        IN,
        filter_num_begin,
        depth,
        stack_num_down,
        stack_num_up,
        patch_size,
        num_heads,
        window_size,
        num_mlp,
        shift_window=shift_window,
        name="swin_unet",
    )

    # Output section
    n_labels = 3
    # OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation="softmax")(X)
    X = keras.layers.UpSampling2D((4, 4))(X)
    OUT = keras.layers.Conv2D(n_labels, (1, 1), activation="softmax")(X)

    # Model configuration
    model = Model(
        inputs=[
            IN,
        ],
        outputs=[
            OUT,
        ],
    )

    model.summary()

    # Optimization
    # <---- !!! gradient clipping is important
    opt = keras.optimizers.Adam(learning_rate=3e-4, clipvalue=1.5)
    # opt = keras.optimizers.SGD(
    #     learning_rate=3e-3, momentum=0.9, weight_decay=0.09, clipvalue=1.0
    # )

    from segmentation_models.losses import (BinaryCELoss, BinaryFocalLoss,
                                            CategoricalCELoss,
                                            CategoricalFocalLoss, DiceLoss,
                                            JaccardLoss)

    # Hyperparam. of the losses
    alpha = 1  # Binary CE
    beta = 0  # Dice
    gamma = 0  # Focal

    # total_loss = alpha * binary_CE_loss + beta * dice_loss + gamma * focal_loss
    loss = DiceLoss()
    model.compile(
        loss=loss,
        optimizer=opt,
        # metrics=[tf.keras.metrics.MeanIoU(num_classes=n_labels, name="mIoU")],
        # callbacks=[tf.keras.callbacks.ProgbarLogger()],
    )

    def input_data_process(input_array):
        """converting pixel vales to [0, 1]"""
        input_array = input_array.astype(np.float32)
        return input_array / 255.0

    def target_data_process(target_array):
        """Converting tri-mask of {1, 2, 3} to three categories."""
        return keras.utils.to_categorical(target_array - 1)

    sample_names = np.array(sorted(glob(filepath + "images/*.jpg")))
    label_names = np.array(sorted(glob(filepath + "annotations/trimaps/*.png")))

    L = len(sample_names)
    ind_all = utils.shuffle_ind(L)

    L_train = int(0.8 * L)
    L_valid = int(0.1 * L)
    L_test = L - L_train - L_valid
    ind_train = ind_all[:L_train]
    ind_valid = ind_all[L_train : L_train + L_valid]
    ind_test = ind_all[L_train + L_valid :]
    print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))

    valid_input = input_data_process(
        utils.image_to_array(sample_names[ind_valid], size=384, channel=3)
    )
    valid_target = target_data_process(
        utils.image_to_array(label_names[ind_valid], size=384, channel=1)
    )

    test_input = input_data_process(
        utils.image_to_array(sample_names[ind_test], size=384, channel=3)
    )
    test_target = target_data_process(
        utils.image_to_array(label_names[ind_test], size=384, channel=1)
    )

    N_epoch = 1024  # number of epoches
    N_batch = 160  # number of batches per epoch
    N_sample = 32  # number of samples per batch

    tol = 0  # current early stopping patience
    max_tol = 4  # the max-allowed early stopping patience
    min_del = 0  # the lowest acceptable loss value reduction
    # path = f"./checkpoint/swin_unet_conv_384_ckpt05/"
    path = f"./checkpoint/swin_unet_conv_384_ckpt{datetime.now().strftime('%m.%d.%H.%M.%S')}/"

    # loop over epoches
    for epoch in range(N_epoch):
        print(f"[{epoch+1}/{N_epoch}]")
        # initial loss record
        if epoch == 0:
            y_pred = model.predict([valid_input])
            record = np.mean(loss(valid_target, y_pred))
            print("\tInitial loss = {}".format(record))

        # loop over batches
        for step in range(N_batch):
            # selecting smaples for the current batch
            ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]

            # batch data formation
            ## augmentation is not applied
            train_input = input_data_process(
                utils.image_to_array(
                    sample_names[ind_train][ind_train_shuffle], size=384, channel=3
                )
            )
            train_target = target_data_process(
                utils.image_to_array(
                    label_names[ind_train][ind_train_shuffle], size=384, channel=1
                )
            )

            # train on batch
            loss_ = model.train_on_batch(
                [
                    train_input,
                ],
                [
                    train_target,
                ],
                reset_metrics=False,
            )
            print(
                f"\r[{step+1}/{N_batch}] loss: {loss_}",
                end="" if step < N_batch - 1 else "\n",
            )

        # print(f"Train mIoU: {loss_[1]}")

        #         if np.isnan(loss_):
        #             print("Training blow-up")

        # ** training loss is not stored ** #

        # epoch-end validation
        # print(f"Train mIoU: {model.metrics_names}")
        if epoch % 10 == 0 and epoch != 0:
            y_pred = model.predict([valid_input])
            print(f"vaild mIoU: {numpy_mean_iou(valid_target, y_pred)}")
            record_temp = np.mean(loss(valid_target, y_pred))
            # ** validation loss is not stored ** #

            # if loss is reduced
            if record - record_temp > min_del:
                print(
                    "Validation performance is improved from {} to {}".format(
                        record, record_temp
                    )
                )
                record = record_temp
                # update the loss record
                tol = 0
                # refresh early stopping patience
                # ** model checkpoint is not stored ** #
                model.save_weights(path)

            # if loss not reduced
            else:
                print("Validation performance {} is NOT improved".format(record_temp))
                tol += 1
                if tol >= max_tol:
                    print("Early stopping")
                    break
                else:
                    # Pass to the next epoch
                    continue

    model.load_weights(path)
    N_sample = 32
    ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]
    ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]

    train_input = input_data_process(
        utils.image_to_array(
            sample_names[ind_train][ind_train_shuffle], size=384, channel=3
        )
    )
    y_pred = model.predict(
        [
            test_input,
        ]
    )
    print(f"Testing set mIoU = {numpy_mean_iou(test_target, y_pred)}")
    print(
        "Testing set cross-entropy loss = {}".format(np.mean(loss(test_target, y_pred)))
    )

    import matplotlib.pyplot as plt

    def ax_decorate_box(ax):
        [j.set_linewidth(0) for j in ax.spines.values()]
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )
        return ax

    from tqdm import tqdm

    i_sample = 453
    for i_sample in tqdm(range(L_test)):
        fig, AX = plt.subplots(1, 4, figsize=(13, (13 - 0.2) / 4))
        plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(
            np.mean(
                test_input[
                    i_sample,
                    ...,
                ],
                axis=-1,
            ),
            cmap=plt.cm.gray,
        )
        AX[1].pcolormesh(test_target[i_sample, ...], cmap=plt.cm.jet)
        AX[2].pcolormesh(y_pred[i_sample, ..., 0], cmap=plt.cm.jet)
        AX[3].pcolormesh(y_pred[i_sample, ..., 2], cmap=plt.cm.jet)

        AX[0].set_title("Original", fontsize=14)
        AX[1].set_title("Pixels belong to the object", fontsize=14)
        AX[2].set_title("Surrounding pixels", fontsize=14)
        AX[3].set_title("Bordering pixels", fontsize=14)

        # plt.show()
        plt.savefig(f"imgs/swin_dice113/{i_sample}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=0)
    args = parser.parse_args()
    g_n = int(args.gpu_id)
    main()
