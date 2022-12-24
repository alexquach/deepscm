import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glow import Flow, Glow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


split_dir = "../../../../../../../../../../../../dgx1nas1/storage/data/jcaicedo/dsae/celebA"
embed_train = np.load(f'{split_dir}/train_features.npy')
embed_valid = np.load(f'{split_dir}/val_features.npy')
embed_test  = np.load(f'{split_dir}/test_features.npy')

split_dir = "../../../../assets/data/celeba"
df_train = pd.read_csv(f'{split_dir}/train_features.csv')
df_valid = pd.read_csv(f'{split_dir}/val_features.csv')
df_test = pd.read_csv(f'{split_dir}/test_features.csv')


# Model stuff
n_flow = 32
n_block = 1
lr = 1e-4
epochs = 10

version = f"_linear_{n_flow}_{n_block}_1e-4_{epochs}"
model_name = "model" + version

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Glow(
    384, n_flow, n_block, affine=True, conv_lu=True
).to(device)
model.load_state_dict(torch.load(f"./{model_name}.pt"))


def generate_forward_embeddings(model, device, dataset, filename):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    with torch.no_grad():
        embeds = []
        for embed in tqdm(dataloader):
            embed = torch.Tensor(embed).to(device)
            embed = embed.reshape(-1, 384)
            log_p_sum, logdet, z_outs = model(embed)
            embeds.append(z_outs[0].cpu().numpy())
        embeds = np.concatenate(embeds)

    # save embeddings
    np.save(f"{filename}.npy", embeds)

def generate_reverse_embeddings(model, device, dataset, filename=None, reconstruct=True):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    with torch.no_grad():
        embeds = []
        for z_embed in tqdm(dataloader):
            z_embed = torch.Tensor(z_embed).to(device)
            z_embed = z_embed.reshape(-1, 384)
            input = model.reverse([z_embed], reconstruct=reconstruct)
            embeds.append(input.cpu().numpy())
        embeds = np.concatenate(embeds)

    # save embeddings
    if filename:
        np.save(f"{filename}.npy", embeds)
    return embeds


train_z = np.load(f'./train_embed{version}.npy').reshape(-1, 384)
valid_z = np.load(f'./valid_embed{version}.npy').reshape(-1, 384)

print("Generating reconstruction")
print(train_z.shape)
train_recon = generate_reverse_embeddings(model, device, train_z, filename=f"./train_reconstruct_{version}")
train_recon = train_recon.reshape(-1, 384)
recon_diff_train = train_recon - embed_train

print(recon_diff_train.shape)
reducer = umap.UMAP(random_state=42)
reducer.fit(embed_train.data)
embedding_valid = reducer.transform(embed_valid)
embedding_train = reducer.transform(embed_train)
# embed_v88 = np.load(f'./v88.npy')
# embedding_v88 = reducer.transform(embed_v88)
# error correction from validation features
embedding_valid_2 = embedding_valid[1:]

embedding_recon_train = reducer.transform(recon_diff_train)

print("MSE Recon Diff Train: ", np.mean(np.square(recon_diff_train)))
# print("MSE Recon Diff Train: ", np.mean(np.square(recon_diff_train)))


colors = [[0, 0, 0], [0, 1, 0]]
descriptions = ["train", "recon"]

plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=[[0, 0, 0]], s=1)
plt.scatter(embedding_recon_train[:, 0], embedding_recon_train[:, 1], c=[[0, 1, 0]], s=1)
plt.gca().set_aspect('equal', 'datalim')

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, descriptions)] 
plt.legend(handles=handles)
plt.title(f'Orig/Recon {version}', fontsize=18)
plt.savefig(f'./orig_recon_{version}.png', dpi=300)