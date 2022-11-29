import torch
import random
import config

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")


def compute_bin_metrics(eval_pred):
    y_pred, y_actual = eval_pred
    y_pred = np.argmax(y_pred, axis=1)
    TP = np.count_nonzero(y_pred * y_actual)
    TN = np.count_nonzero((y_pred - 1) * (y_actual - 1))
    FP = np.count_nonzero(y_pred * (y_actual - 1))
    FN = np.count_nonzero((y_pred - 1) * y_actual)
    precision = np.float64(TP)/(TP+FP)
    recall = np.float64(TP)/(TP+FN)
    accuracy = np.float64((TP+TN))/(TP+TN+FP+FN)
    f1 = (2*precision*recall)/(precision+recall)
    return {'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1}


def softmax(x, axis=-1):
    """Traditional softmax activation"""
    eX = np.exp(x)
    return eX/eX.sum(axis=axis, keepdims=True)


def train_step(model, batch_data, optimizer, loss_fn):
    logits = model(batch_data['input_ids'].to(device))
    loss = loss_fn(logits, torch.eye(2)[batch_data['labels']].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred = softmax(logits.cpu().detach().numpy())
    return y_pred, loss.item()


def test_step(model, batch_data, optimizer, loss_fn):
    logits = model(batch_data['input_ids'].to(device))
    loss = loss_fn(logits, torch.eye(2)[batch_data['labels']].to(device))
    y_pred = softmax(logits.cpu().detach().numpy())
    return y_pred, loss.item()


def train(model, train_dataset, test_dataset, n_epochs=30, batch_size=64, optimiz_metric='f1', epoch_callback=None,
          learning_rate=1e-3, weight_decay=0, out_model_file=None, random_seed=0):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_metric = 0
    train_res = DNNTrainingResults()
    for epoch in range(n_epochs):
        model.train()
        for train_batch in train_loader:
            preds, loss = train_step(model, train_batch, optimizer, loss_fn)
            train_res.add_batch_results(preds, loss, epoch, 'train')
        train_metrics = train_res.compute_epoch_metrics(epoch, 'train', train_dataset.labels)
    
        model.eval()
        for test_batch in test_loader:
            preds, loss = test_step(model, test_batch, optimizer, loss_fn)
            train_res.add_batch_results(preds, loss, epoch, 'eval')
        eval_metrics = train_res.compute_epoch_metrics(epoch, 'eval', test_dataset.labels)
        
        if epoch_callback is not None:
            epoch_callback(epoch, train_res)

        train_batch, test_batch = None, None  # Release memory
        if eval_metrics[optimiz_metric] > best_metric:
            if out_model_file is not None:
                torch.save(model.state_dict(), out_model_file)
            best_metric = eval_metrics[optimiz_metric]
    return train_res


def predict_docs_batch(docs, model, tokenizer, batch_size=8, hf_model=False):
    model.eval()
    dataset = HFDataset(tokenizer(docs, truncation=True, padding=True))
    data_loader = DataLoader(dataset, batch_size=batch_size)
    y_pred = []
    for batch_data in data_loader:
        batch_data['input_ids'] = batch_data['input_ids'].to(device)
        if hf_model:
            if 'token_type_ids' in batch_data:
                batch_data['token_type_ids'] = batch_data['token_type_ids'].to(device)
            batch_data['attention_mask'] = batch_data['attention_mask'].to(device)
            logits = model(**batch_data)['logits'].detach().cpu().numpy()
        else:
            logits = model(batch_data['input_ids']).detach().cpu().numpy()
        y_pred.append(softmax(logits))
    return np.vstack(y_pred)    


def _embed_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedder():
    return Embedder('files/all-MiniLM-L6-v2')

def get_tokenizer():
    return AutoTokenizer.from_pretrained("files/bert-tokenizer", use_fast=True)

class DNNTrainingResults():
    """Class to easily store and summarize training results"""
    def __init__(self):
        self.results = {}  
        self.summary_results = {} 

    def add_batch_results(self, preds, loss, epoch, phase):
        if epoch not in self.results.keys():
            self.results[epoch] = {}
        if phase not in self.results[epoch].keys():
            self.results[epoch][phase] = {'preds': [], 'losses': []}
        self.results[epoch][phase]['preds'].append(preds)
        self.results[epoch][phase]['losses'].append(loss)
    
    def compute_epoch_metrics(self, epoch, phase, y_actual):
        if epoch not in self.summary_results.keys():
            self.summary_results[epoch] = {}
        epres = self.results[epoch][phase]
        preds, losses = np.vstack(epres['preds']), np.array(epres['losses'])
        metrics = compute_bin_metrics((preds, y_actual))
        metrics["loss"] = np.mean(losses)
        self.summary_results[epoch].update({phase: metrics})
        return metrics

    def summary_df(self, operat=np.mean):
        return pd.DataFrame([{f"{phase}_{name}": val for phase, metrics in epoch_data.items() \
                              for name, val in metrics.items()} \
                             for epoch_data in self.summary_results.values()], index=self.summary_results.keys())


class HFDataset(torch.utils.data.Dataset):
    """Used for fine tuning of hugging face models"""
    def __init__(self, encodings, labels=None):
        self.encodings, self.labels = encodings, labels

    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class CNNModel(nn.Module):
    def __init__(self, vocab_size, num_classes=2, filter_sizes=[3, 4, 5], embedding_size=100, 
                 num_filters=128, drop_proba=0.5, pretrained_embeddings=None):
        super(CNNModel, self).__init__()
        if pretrained_embeddings is not None:
            pretrained = torch.FloatTensor(pretrained_embeddings)
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, num_filters, (filter_size, embedding_size), stride=1, padding=0) \
                                          for filter_size in filter_sizes])
        self.dropout = nn.Dropout(drop_proba)
        self.label = nn.Linear(len(filter_sizes)*num_filters, num_classes)

    def forward(self, input_X):
        X = self.word_embeddings(input_X)
        X = X.unsqueeze(1)

        conv_out = []
        for conv_layer in self.conv_layers:  # Do convolution, relu and max pooling
            activation  = F.relu(conv_layer(X).squeeze(3))
            conv_out.append(F.max_pool1d(activation, activation.size()[2]).squeeze(2))

        all_conv_out = torch.cat(conv_out, 1)
        drop_res = self.dropout(all_conv_out)
        logits = self.label(drop_res)
        return logits


class Embedder:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model_path = model_path
        self.model.to(device)
    
    
    def encode(self, texts, normalize_embeddings=False, batch_size=32):
        self.model.eval()
        dataset = HFDataset(self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt'))
        data_loader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        for batch_data in data_loader:
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            embeddings.append(self._forward_pass(batch_data))
        embeddings = torch.cat(embeddings)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().detach().numpy()
    

    def _forward_pass(self, input_data):
        with torch.no_grad():
            model_output = self.model(**input_data, return_dict=True)
        
        return _embed_mean_pooling(model_output, input_data['attention_mask'])