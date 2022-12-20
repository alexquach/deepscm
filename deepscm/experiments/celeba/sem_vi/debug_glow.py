import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from math import log, sqrt, pi

from torch.utils.data import DataLoader
from glow import Flow, Glow
import torch
from torch import optim

split_dir = "../../../../assets/data/celeba"
# split_dir = "assets/data/celeba"
# split_dir = "../../../../../../../../dgx1nas1/storage/data/jcaicedo/dsae/celebA"
# celeba_train = CelebaEmbedDataset(npy_path=, csv_path=f'{split_dir}/train_features.csv')
embed_train = np.load(f'{split_dir}/train_features.npy')
embed_valid = np.load(f'{split_dir}/val_features.npy')
# embed_test  = np.load(f'{split_dir}/test_features.npy')deepscm/experiments/celeba/sem_vi/umap_new.ipynb

def calc_loss(log_p, logdet, num_features, n_bins):
    # log_p = calc_log_p([z_list])

    loss = -log(n_bins) * num_features
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * num_features)).mean(),
        (log_p / (log(2) * num_features)).mean(),
        (logdet / (log(2) * num_features)).mean(),
    )

def train(dataset, model, optimizer, epochs, lr, device):
    with tqdm(enumerate(iter(dataset))) as pbar:
        for i, embed in pbar:
            # embed = next(dataset)
            embed = embed.to(device)
            embed_size = embed.shape[1]
            embed = embed.reshape(-1, 384, 1, 1)

            # dequantization
            # log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
            log_p, logdet, _ = model(embed)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, 384, 256)

            model.zero_grad()
            loss.backward()
            
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

n_flow = 32
n_block = 1
lr = 1e-4
epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

is_train = False
version = "_base"
model_name = "model" + version

def generate_embeddings(model, device, dataset, filename):

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    with torch.no_grad():
        embeds = []
        for embed in tqdm(dataloader):
            embed = torch.Tensor(embed).to(device)
            embed = embed.reshape(-1, 384, 1, 1)
            log_p_sum, logdet, z_outs = model(embed)
            embeds.append(z_outs[0].cpu().numpy())
        embeds = np.concatenate(embeds)

    # save embeddings
    np.save(f"{filename}.npy", embeds)

if is_train:
    train_dataloader = DataLoader(embed_train, batch_size=64, shuffle=True)

    model = Glow(
        384, n_flow, n_block, affine=True, conv_lu=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(train_dataloader, model, optimizer, epochs, lr, device)
    torch.save(
        model.state_dict(), f"./{model_name}.pt"
    )

else:
    model = Glow(
        384, n_flow, n_block, affine=True, conv_lu=True
    ).to(device)
    # model.load_state_dict(torch.load(f"./deepscm/experiments/celeba/sem_vi/{model_name}.pt"))
    model.load_state_dict(torch.load(f"./{model_name}.pt"))


generate_embeddings(model, device, embed_train, "train_embed" + version)
generate_embeddings(model, device, embed_valid, "valid_embed" + version)