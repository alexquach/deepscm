import numpy as np
import os
import pandas as pd
import pyro
import torch

from pyro.distributions import Gamma, Normal, TransformedDistribution
from pyro.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform

from tqdm import tqdm

from datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from datasets.morphomnist.transforms import SetThickness, ImageMorphology


def get_intensity(img):
    threshold = 0.5

    img_min, img_max = img.min(), img.max()
    mask = (img >= img_min + (img_max - img_min) * threshold)
    avg_intensity = img[mask].mean()

    return avg_intensity


def model(n_samples=None, scale=2.):
    with pyro.plate('observations', n_samples):
        thickness = pyro.sample('thickness', Gamma(10., 5.))

        loc = (thickness - 2.5) * 2

        transforms = ComposeTransform([SigmoidTransform(), AffineTransform(64, 195)])

        intensity = pyro.sample('intensity', TransformedDistribution(Normal(loc, 0.5), transforms))

    return thickness, intensity


def gen_dataset(args, train=True):
    pyro.clear_param_store()
    images, labels, _ = load_morphomnist_like(args.data_dir, train=train)

    if args.digit_class is not None:
        mask = (labels == args.digit_class)
        images = images[mask]
        labels = labels[mask]

    n_samples = len(images)
    with torch.no_grad():
        thickness, intensity = model(n_samples, scale=args.scale)

    metrics = pd.DataFrame(data={'thickness': thickness, 'intensity': intensity})

    for n, (thickness, intensity) in enumerate(tqdm(zip(thickness, intensity), total=n_samples)):
        morph = ImageMorphology(images[n], scale=16)
        tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))

        avg_intensity = get_intensity(tmp_img)

        mult = intensity.numpy() / avg_intensity
        tmp_img = np.clip(tmp_img * mult, 0, 255)

        images[n] = tmp_img

    # TODO: do we want to save the sampled or the measured metrics?

    save_morphomnist_like(images, labels, metrics, args.out_dir, train=train)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/vol/biomedic/users/dc315/mnist/original/', help="Path to MNIST (default: %(default)s)")
    parser.add_argument('-o', '--out-dir', type=str, help="Path to store new dataset")
    parser.add_argument('-d', '--digit-class', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="digit class to select")
    parser.add_argument('-s', '--scale', type=float, default=2., help="scale of logit normal")

    args = parser.parse_args()

    print(f'Generating data for:\n {args.__dict__}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        print(f'Generated data for:\n {args.__dict__}', file=f)

    print('Generating Training Set')
    print('#######################')
    gen_dataset(args, True)

    print('Generating Test Set')
    print('###################')
    gen_dataset(args, False)
