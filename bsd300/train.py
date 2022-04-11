'''
Train and test BSD300 dataset with limited network complexity
'''

# %%
import random
import os
import sys
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import imageio
import matplotlib.pyplot as plt
import glob
import pandas as pd


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--working_dir', default='/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/bsd300_denoising/'
    )
    parser.add_argument('--train_dir', default='BSDS300/images/train')
    parser.add_argument('--test_dir', default='BSDS300/images/test')

    parser.add_argument('--loss_norm', type=int, default=1)
    parser.add_argument('--noise_std_1', type=float, default=25)
    parser.add_argument('--noise_std_2', type=float, default=25)

    parser.add_argument('--device', default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def load_imgs(filename_dir: str):
    filenames = glob.glob(os.path.join(filename_dir, '*'))
    imgs = []
    for filename in filenames:
        try:
            img = imageio.imread(filename)
            if img.shape[0] > img.shape[1]:
                img = img.transpose([1, 0, 2])
            imgs.append(img)
        except Exception:
            pass
    return np.array(imgs).astype(np.float32)


def process_img_for_plot(img):
    img = (img + 1) * 127.5
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


# %%
def load_bsd300_and_add_noise(
    train_dir: str,
    test_dir: str,
    noise_std_1: float,
    noise_std_2: float,
    seed=0,
    plot=False
):
    rng = np.random.default_rng(seed)

    train_y = load_imgs(train_dir)
    train_n1 = rng.normal(0, noise_std_1, train_y.shape)
    train_n2 = rng.normal(0, noise_std_2, train_y.shape)
    train_x1 = (train_y + train_n1) / 127.5 - 1
    train_x2 = (train_y + train_n2) / 127.5 - 1
    train_y = train_y / 127.5 - 1

    test_y = load_imgs(test_dir)
    test_n = rng.normal(0, noise_std_1, test_y.shape)
    test_x = (test_y + test_n) / 127.5 - 1
    test_y = test_y / 127.5 - 1

    if plot:
        plt.figure(figsize=[16, 12])
        plt.subplot(221)
        plt.imshow(process_img_for_plot(train_x1[0]))
        plt.subplot(222)
        plt.imshow(process_img_for_plot(train_x2[0]))
        plt.subplot(223)
        plt.imshow(process_img_for_plot(test_x[0]))
        plt.subplot(224)
        plt.imshow(process_img_for_plot(test_y[0]))

    return train_x1, train_x2, train_y, test_x, test_y


# %%
def build_model(args, input_shape):
    K.clear_session()

    input_layer = tf.keras.Input(input_shape)
    x = input_layer
    for i in range(args.depth):
        x = tf.keras.layers.Conv2D(args.channels, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    output_layer = tf.keras.layers.Conv2D(input_shape[-1], 1, padding='same')(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

    optimizer = tf.optimizers.Adam(args.lr)

    if args.loss_norm == 2:
        loss = tf.keras.losses.MeanSquaredError()
    elif args.loss_norm == 1:
        loss = tf.keras.losses.MeanAbsoluteError()
    else:
        raise ValueError('loss_norm must be either 1 or 2')

    model.compile(optimizer, loss)

    return model


# %%
def predict_model(
    model: tf.keras.Model,
    x: np.array,
    y: np.array,
):
    pred = model.predict(x)

    pred = (pred + 1) * 127.5
    x = (x + 1) * 127.5
    y = (y + 1) * 127.5

    return pred, x, y


def l2_loss(x, y):
    return np.mean(np.sqrt(np.mean((x - y)**2, (1, 2, 3))))


def l1_loss(x, y):
    return np.mean(np.abs(x - y))


def psnr(x, y):
    mse = np.mean((x - y)**2, (1, 2, 3))
    return np.mean(10 * np.log10(255 * 255 / mse))


def evaluation(
    model: tf.keras.Model,
    train_x: np.array,
    train_y: np.array,
    test_x: np.array,
    test_y: np.array
) -> pd.DataFrame:
    train_p, train_x, train_y = predict_model(model, train_x, train_y)
    test_p, test_x, test_y = predict_model(model, test_x, test_y)

    loss_func = {
        'PSNR': psnr,
        'L2': l2_loss,
        'L1': l1_loss
    }

    df = []
    for name in loss_func:
        loss = loss_func[name]
        df.append({
            'Loss': name,
            'Data': 'Pred',
            'Train': loss(train_p, train_y),
            'Test': loss(test_p, test_y)
        })

        df.append({
            'Loss': name,
            'Data': 'x',
            'Train': loss(train_x, train_y),
            'Test': loss(test_x, test_y)
        })
    df = pd.DataFrame(df)

    return df


# %%
def main(args):
    # set seed
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # load dataset
    train_x1, train_x2, train_y, test_x, test_y = load_bsd300_and_add_noise(
        os.path.join(args.working_dir, args.train_dir),
        os.path.join(args.working_dir, args.test_dir),
        args.noise_std_1,
        args.noise_std_2,
        args.seed,
        args.debug
    )

    # model
    model = build_model(args, train_x1.shape[1:])

    model.fit(
        train_x1,
        train_x2,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(test_x, test_y),
        verbose=(1 if args.debug else 2)
    )

    # setting output
    output_dir = os.path.join(
        args.working_dir,
        'depth_{0}_ch_{1}_epoch_{2}'.format(args.depth, args.channels, args.epochs)
    )
    filename = 'norm_{0}_std1_{1}_std2_{2}_seed_{3}'.format(
        args.loss_norm, args.noise_std_1, args.noise_std_2, args.seed
    )
    os.makedirs(output_dir, exist_ok=True)

    # save model
    model.save(os.path.join(output_dir, filename + '.h5'))

    # evaluate and save
    df = evaluation(model, train_x1, train_y, test_x, test_y)
    df.to_csv(os.path.join(output_dir, filename + '.csv'), index=False)

    return df


# %%
if __name__ == '__main__':
    args = get_args()
    df = main(args)
