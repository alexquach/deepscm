models, embeddings, and visualizations are on my dgx

`debug_glow.py` (unfortunately named)
* script to train models and generate latent (z) embeddings
* hyperparameters like number of blocks, epochs are in the code

`eval_model.ipynb`
* file inputs (are the saved files from `debug_glow.py`):
    * model: `{model_name}.pt`
    * train embeddings: `train_embed_{model_version}.npy`
    * validation embeddings: `valid_embed_{model_version}.npy`
* hyperparameters like number of blocks, epochs need to be adjusted
* Reconstructions are generated with `generate_reverse_embeddings`
* Then visualizes original + reconstructions on same UMAP
* MSE is calculated on pre-UMAP transformed original + reconstructed features


`eval_model.py`
* script version; Slightly out of date, should be replaced with snippets from `eval_model.ipynb`