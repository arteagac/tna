import json
import numpy as np
import matplotlib.pyplot as plt

from texpl import compute_top_words
from utils import read_file, write_file
from string import Template


def plot_train_loss(rdf, out_file=None, dpi=300):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.array(rdf.index) + 1
    ax.plot(epochs, rdf['train_loss'], color='#ff8c5e', label='train_loss')
    ax.plot(epochs, rdf['eval_loss'], color='#ff8c5e', linestyle='--', label='eval_loss')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Training Loss", "Validation Loss"])
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=dpi)

    
def cluster_visualization(res_clusters, corpus, scores, uq_ids, template_file, out_file):
    clusters_json = []
    scores = np.array(scores)
    for c in np.unique(res_clusters):
        in_cluster = np.where(res_clusters==c)[0]
        top_words = compute_top_words([corpus[i] for i in in_cluster], n_top=4)
        clusters_json.append(
            {'name': ', '.join(top_words), 'score': round(float(scores[in_cluster].mean()), 3),
             'children': [{
                 'name': corpus[i],
                 'score': round(float(scores[i]), 3),
                 'id': uq_ids[i]
             } for i in in_cluster]})
    
    clusters_json = {'name': '', 'children': clusters_json}
    d3_src = Template(read_file(template_file))

    html_src = d3_src.substitute({'python_data': json.dumps(clusters_json, indent=4)})
    write_file(out_file, html_src)
    

def plot_num_clusters(dis_thres, n_clusters, size=(6, 4), out_file=None, dpi=300):
    fig, ax = plt.subplots(figsize=size)
    bp = ax.bar([f"{e:.2f}" for e in dis_thres], n_clusters, color='#ffce72')
    ax.bar_label(bp)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of clusters")
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=dpi)


def plot_num_phrases(thres_vals, n_phrases, size=(6, 4), out_file=None, dpi=300):
    fig, ax = plt.subplots(figsize=size)
    bp = ax.bar([f"{e:.2f}" for e in thres_vals], n_phrases, color='#ff8c5e')
    ax.bar_label(bp)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Standard deviations")
    plt.ylabel("Number of phrases")
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=dpi)
