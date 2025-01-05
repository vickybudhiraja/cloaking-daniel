from utils.common import get_ends
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import tqdm
import os


class P3MaskGeneration:
    NUM_EPOCHS = 50
    EPSILON = 16.
    LEARNING_RATE = 1e-3
    IMAGE_SHAPE = (112, 112, 3)
    BATCH_SIZE = 1

    def __init__(self, bottleneck_models, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, epsilon=EPSILON, image_shape=IMAGE_SHAPE):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.step = self.learning_rate * 255
        self.bottleneck_models = bottleneck_models
        self.image_shape = image_shape
        self.batch_size = batch_size
        np.random.seed(2023)
        self.umask = np.random.uniform(-self.step, self.step, tuple([1] + list(image_shape)))

    def calculate_feature_space_loss(self, tape, src_img, adv_img):
        feature_distances = []
        for bottleneck_model in self.bottleneck_models:
            if tape is not None:
                try:
                    tape.watch(bottleneck_model.model.variables)
                except AttributeError:
                    tape.watch(bottleneck_model.variables)
            # get the respective feature space reprs.
            bottleneck_src = bottleneck_model(src_img)
            # compute the arccosine distance
            bottleneck_adv = bottleneck_model(adv_img)
            feature_distance = tf.math.acos(tf.reduce_sum(bottleneck_src * bottleneck_adv, axis=-1))
            feature_distances.append(feature_distance)
        return feature_distances

    def calculate_dissimilarity_loss(self, source_raw, source_mod_raw):
        ssims = tf.image.ssim(source_raw, source_mod_raw, max_val=255.0)
        dist = (1.0 - tf.stack(ssims)) / 2.0
        return dist[0], ssims[0]

    def compute_loss(self, tape, src_img, adv_img):
        """ Compute input space + feature space loss.
        """
        feature_space_losses = self.calculate_feature_space_loss(tape, src_img, adv_img)
        dissimilarity_loss, ssim = self.calculate_dissimilarity_loss(src_img, adv_img)
        return feature_space_losses, dissimilarity_loss, ssim

    def compute(self, protectee_train_imgs, protectee_train_bbox,
                output_dir, l_threshold=0.03, random_seed=2023, margin=30):
        # Generate non-padding area masks
        protectee_train_bbox = np.asarray([protectee_train_bbox[i] for i in range(len(protectee_train_imgs))])

        # Set the random seed for per-epoch randomization
        random.seed(random_seed)

        # Initial weight for the SSIM component
        lambda_dsim = 1.
        for epoch in range(self.num_epochs):
            # Shuffle training images
            indices = list(range(len(protectee_train_imgs)))
            random.shuffle(indices)
            protectee_train_imgs = protectee_train_imgs[indices]
            protectee_train_bbox = protectee_train_bbox[indices]
            epoch_loss, epoch_loss_feat, epoch_loss_dsim, epoch_ssim = [], [], [], []
            pbar = tqdm.tqdm(range(0, len(protectee_train_imgs), self.batch_size))
            for it in pbar:

                batch_loss, batch_loss_feat, batch_loss_dsim, batch_ssim = [], [], [], []
                grads = []
                for offset in range(min(self.batch_size, len(protectee_train_imgs) - it)):
                    src_img = protectee_train_imgs[it+offset:it+offset+1]  # (1, 112, 112, 3)
                    # The shape of the face BEFORE box-preprocessing
                    face_shape = (protectee_train_bbox[it+offset][3]-protectee_train_bbox[it+offset][1],
                                  protectee_train_bbox[it+offset][2]-protectee_train_bbox[it+offset][0])
                    # The shape of the boxed face BEFORE resizing to (112, 112)
                    source_shape = (max(face_shape) + margin, max(face_shape) + margin)
                    # Coordinates where the face is put on the box
                    start1, end1 = get_ends(source_shape[0], face_shape[0])
                    start2, end2 = get_ends(source_shape[0], face_shape[1])

                    bmask = np.full(shape=(1, *source_shape, 3), fill_value=0.)
                    bmask[0, start1:end1, start2:end2, :] = 1.

                    x_benign_tf = tf.constant(src_img, dtype=np.float32)
                    bmask_tf = tf.constant(bmask, dtype=np.float32)
                    umask_tf = tf.Variable(self.umask, dtype=np.float32)

                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(umask_tf)
                        # Generate the protected face
                        x_adv_tf = tf.clip_by_value(x_benign_tf + umask_tf, clip_value_min=0., clip_value_max=255.)
                        # Scale up to the original boxed size
                        x_benign_scaled_tf = tf.image.resize(x_benign_tf, source_shape)
                        x_adv_scaled_tf = tf.image.resize(x_adv_tf, source_shape)
                        # Remove UMask on padded area
                        x_adv_cleaned_scaled_tf = x_adv_scaled_tf * bmask_tf + x_benign_scaled_tf * (1 - bmask_tf)
                        # Scale down to (112, 112) for processing by the FRs
                        x_benign_downscaled_tf = tf.image.resize(x_benign_scaled_tf, (112, 112))
                        x_adv_cleaned_downscaled_tf = tf.image.resize(x_adv_cleaned_scaled_tf, (112, 112))
                        # Compute loss and generate gradients
                        feature_space_losses, dissimilarity_loss, ssim = self.compute_loss(tape,
                                                                                           x_adv_cleaned_downscaled_tf,
                                                                                           x_benign_downscaled_tf)
                        feature_space_loss = tf.reduce_sum(feature_space_losses)
                        loss = -feature_space_loss + lambda_dsim * tf.maximum(dissimilarity_loss - l_threshold, 0.0)
                        grad = tape.gradient(loss, [umask_tf])[0].numpy()

                        batch_loss.append(loss.numpy())
                        batch_loss_feat.append(feature_space_loss.numpy())
                        batch_loss_dsim.append(dissimilarity_loss.numpy())
                        batch_ssim.append(ssim.numpy())
                        grads.append(grad)
                grad = np.mean(np.concatenate(grads, axis=0), axis=0, keepdims=True)
                grad_sign = np.sign(grad)

                # Update UMask
                self.umask = np.clip(self.umask - self.step * grad_sign, -self.epsilon, self.epsilon)

                # Compute statistics for the current mini-batch
                epoch_loss.append(np.mean(batch_loss))
                epoch_loss_feat.append(np.mean(batch_loss_feat))
                epoch_loss_dsim.append(np.mean(batch_loss_dsim))
                epoch_ssim.append(np.mean(batch_ssim))

                # Update the lambda dynamically
                if epoch_loss_dsim[-1] <= l_threshold * 0.90 and lambda_dsim >= 1. / 129:
                    lambda_dsim /= 2
                elif epoch_loss_dsim[-1] >= l_threshold * 1.10 and lambda_dsim <= 129.:
                    lambda_dsim *= 2
                elif l_threshold * 0.90 < epoch_loss_dsim[-1] < l_threshold * 1.10:
                    lambda_dsim = 1.0

                pbar.set_description(
                    'Epoch %d / %d - [L↓: %.4f] FEAT↑: %.4f | DSIM↓: %.4f | SSIM: %.4f' % (
                        epoch + 1, self.num_epochs,
                        float(np.mean(epoch_loss)),
                        float(np.mean(epoch_loss_feat)),
                        float(np.mean(epoch_loss_dsim)),
                        float(np.mean(epoch_ssim))))
            # write umask - npy
            os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
            fpath_umask = os.path.join(output_dir, 'npy', 'epoch_%04d.npy' % epoch)
            np.save(fpath_umask, self.umask[0])

            # write umask - jpg
            os.makedirs(os.path.join(output_dir, 'jpg'), exist_ok=True)
            umask_jpg = (self.umask[0] - self.umask.min()) / (self.umask.max() - self.umask.min())
            Image.fromarray(np.uint8(umask_jpg * 255.)).save(os.path.join(output_dir, 'jpg', 'epoch_%04d.jpg' % epoch))
        return self.umask[0]
