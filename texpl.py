import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def rolling(a, op=np.mean, ws=5):
    return np.array([op(a[i-min(i, ws):i]) for i in range(1, len(a) + ws)])


def sliding_window_pert(words, ws=5):
    return [' '.join(list(words[np.arange(i, i + ws)])) for i in range(len(words) - ws + 1)]


def texpl_scores(doc, ws=5, predict_fn=None, expl_class=1):
    text_arr = np.array(doc.split())
    pert_text = sliding_window_pert(text_arr, ws=ws)
    preds = predict_fn(pert_text)[:, expl_class]
    mean_scores = rolling(preds, op=np.mean, ws=ws)
    max_scores = rolling(preds, op=np.max, ws=ws)
    return {'mean_scores': mean_scores, 'max_scores': max_scores}


def texpl_peakdet(scores_dict, sd_threshold=1.2, min_expl_len=2):
    mean_scores, max_scores = scores_dict['mean_scores'], scores_dict['max_scores']
    scores = mean_scores
    peaks = np.abs(scores) > np.mean(scores) + sd_threshold*np.std(scores)
    idx_peaks = np.where(peaks)[0]
    explan_idx = np.split(idx_peaks, np.where(np.diff(idx_peaks) != 1)[0] + 1)
    explan_idx = [expl for expl in explan_idx if len(expl) > min_expl_len and any(max_scores[expl] > .5)]
    explan_scores = [np.mean(scores[x]) for x in explan_idx]
    return explan_idx, explan_scores


def texpl_peakdet_process(res_scores, docs, uq_ids, sd_threshold=1.2):
    res_explan = []
    for rs in res_scores:
        explan_idx, explan_scores = texpl_peakdet(rs['scores'], sd_threshold=sd_threshold)
        text_arr = np.array(docs[rs['idx']].split())
        explan_text = [' '.join(text_arr[x]) for x in explan_idx]
        res_explan.append({'text': explan_text, 'scores': explan_scores, 'uq_ids': [uq_ids[rs['idx']]]*len(explan_idx)})
    return res_explan


def texpl_scores_all(idx_list, docs, ws=5, predict_fn=None, progress_cb=None):
    res = []
    for i, idx in enumerate(idx_list):
        res.append({'idx': idx, 'scores': texpl_scores(docs[idx], ws=ws, predict_fn=predict_fn)})
        if progress_cb is not None:
            progress_cb(i+1)
    return res


def embed_corpus(corpus, embedder, normalize=True, rm_stop_words=True):
    if rm_stop_words:
        corpus = [' '.join([w for w in doc.split() if w not in ENGLISH_STOP_WORDS]) for doc in corpus]
    corpus_embeddings = embedder.encode(corpus, normalize_embeddings=normalize)
    return corpus_embeddings


def cluster_embeddings(corpus_embeddings, distance_threshold=1.25, distance_metric='euclidean', linkage='average'):
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                               affinity=distance_metric, linkage=linkage)
    clustering_model.fit(corpus_embeddings)
    res_clusters = clustering_model.labels_
    return res_clusters


def compute_top_words(sentences, n_top=3):
    from collections import Counter
    words = [w.lower() for s in sentences for w in re.split('[;.,\s]+', s) if len(w) > 1]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and w.isalpha()]
    counter = Counter(words)
    return [x[0] for x in counter.most_common(n_top)]


def peak_det_sensit(res_scores):
    thres_vals = np.array([.75, 1, 1.25, 1.5])
    n_phrases =  []
    for sd_thres in thres_vals:
        explan_idx = [texpl_peakdet(rs['scores'], sd_threshold=sd_thres)[0] for rs in res_scores]
        explan_idx = [y for x in explan_idx for y in x]
        n_phrases.append(len(explan_idx))
    return thres_vals, n_phrases


def clustering_sensit(embeddings):
    dis_thres = [.75, 1, 1.25, 1.5]
    n_clusters = []
    for i in dis_thres:
        res_clusters = cluster_embeddings(embeddings, distance_threshold=i)
        n_clusters.append(len(np.unique(res_clusters)))
    return dis_thres, n_clusters

