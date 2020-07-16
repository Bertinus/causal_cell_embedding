import numpy as np
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot(outputs_tsne, fig, col, size, line_true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(outputs_tsne):
        ax.scatter(point[0], point[1], s=size, c=col[line_true_labels[i]])


def plotClusters(tqdm, outputs_emb, line):
    tqdm.write('Start plotting using TSNE...')

    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    outputs_tsne = tsne.fit_transform(outputs_emb)

    # Plot figure [Here I consider line as true label....Please edit it if this is wrong]
    fig = plt.figure()
    plot(outputs_tsne, fig, ['red', 'green'], 4, line)
    fig.savefig("plot.png")
    tqdm.write("Finished plotting")


def visualize(model, device, train_loader):
    model.train()

    for batch_idx, data in enumerate(train_loader):

        x, fingerprint, compound, line = data
        x = x.to(device)
        fingerprint = fingerprint.to(device)

        # Expected to return a dictionary of outputs.
        outputs_emb = model.forward(x, fingerprint, compound, line)

	plotClusters(tqdm, np.load(outputs_emb), line)



