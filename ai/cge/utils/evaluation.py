from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch


def tsne(mdl, dataloader):
    """Use t-sne to plot the latent space of the model."""
    embeddings, labels = [], []

    for i, (X, S, y, box) in enumerate(dataloader):

        # Sample 5 batches for visualization.
        if i == 5:
            break

        X = X.to(mdl.device)
        try:
            emb = model.X.embed(X)  # Multimodal models.
        except:
            emb = model.embed(X)  # Unimodal models.

        if type(emb) == tuple:
            emb = emb[0]  # First is always assumed to be the latent.

        embeddings.append(emb.detach().cpu().numpy())
        labels.append(y.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # Settings optimized for short run-time.
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2, random_state=0, perplexity=30,
                learning_rate=100, n_iter=5000)

    emb = tsne.fit_transform(pca.fit_transform(embeddings))

    # Save the t-sne embedding to a plot (colour-coded by class).
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels)

    return plt
