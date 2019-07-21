import os
import time
import logging
import functools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiLabelSoftMarginLoss, BCEWithLogitsLoss
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler, WeightedRandomSampler)
from tqdm import tqdm, trange

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from utils import pre_processing


class PretrainedBert:
    def __init__(self, model_name='bert-base-multilingual-cased', batch_size=8,
                 gradient_accumulation_steps=8, n_epochs=3, hierarchical=False,
                 label_hierarchy=None, loss='bce'):
        self.batch_size = batch_size
        self.label_map = {k: v for v, k in enumerate(['O', 'X', '[CLS]', '[SEP]'], 1)}
        self.model = None
        self.model_path = None
        self.tokenizer = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.hierarchical = hierarchical
        self.label_hierarchy = label_hierarchy
        self.post_epoch_hook = None
        if loss not in {'bce', 'multilabel-softmargin'}:
            raise ValueError('loss needs to be one of ["bce", "multilabel-softmargin"]')
        self.loss = loss

    def label2idx(self, label):
        if label is None:
            return self.label_map['O']

        if label not in self.label_map:
            self.label_map.update({label: len(self.label_map) + 1})  # pad label is not in the map
        return self.label_map[label]

    def tokenize_one_(self, msg: str, tokenizer: BertTokenizer = None):
        bert_tokens = []
        wp_starts = []
        truncated = False
        if tokenizer is None:
            raise Exception('Tokenizer can not be None.')

        tokens = pre_processing.tokenise(f'{msg["title"]}. {msg["body"]}',
                                         lowercase=False,
                                         simple=True,
                                         remove_stopwords=False)
        for i_token, token_str in enumerate(tokens):
            skip_token = False
            wordpieces = tokenizer.tokenize(token_str)

            if not wordpieces:
                # this mainly happens for strange unicode characters
                token_str = '[UNK]'
                wordpieces = tokenizer.tokenize(token_str)
                skip_token = True

            if len(bert_tokens) + len(wordpieces) > 510:
                # bert model is limited to 512 tokens
                truncated = True
                break

            if not skip_token:
                wp_starts.append(len(bert_tokens) + 1)  # first token is [CLS]

            bert_tokens.extend(wordpieces)
        bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']

        assert len(bert_tokens) <= 512, f'{len(bert_tokens)} > 512'

        bert_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        return bert_ids, wp_starts, truncated

    def tokenize(self, msgs):
        n_truncated = 0
        input_tokens = []
        input_masks = []

        tokenizer = BertTokenizer.from_pretrained(self.model_name,
                                                  do_lower_case=False)
        f = functools.partial(self.tokenize_one_, tokenizer=tokenizer)
        for results in map(f, msgs):
            bert_ids, wp_starts, truncated = results
            n_truncated += truncated

            # 0 maps to the [PAD] token in BertTokenizer
            ids = np.zeros((512,), dtype=np.int64)
            ids[:len(bert_ids)] = bert_ids
            input_tokens.append(ids)

            # the attention mask is over all non PAD tokens
            attn_mask = np.zeros(ids.shape, dtype=np.int64)
            attn_mask[wp_starts] = 1
            input_masks.append(attn_mask)
        return np.asarray(input_tokens), np.asarray(input_masks)

    def fit(self, X, y, dev=None):
        tokens, masks = self.tokenize(X)
        tokens = torch.LongTensor(tokens)
        masks = torch.LongTensor(masks)
        y = torch.FloatTensor(y)
        train_data = TensorDataset(tokens, y, masks)

        if self.hierarchical:
            class_weights = y.sum(dim=0)
            for lvl, idx in self.label_hierarchy.items():
                class_weights[idx] = (lvl + 1) - (class_weights[idx] / class_weights[idx].sum())
            instance_weights = (y * class_weights).mean(dim=1)
        else:
            class_weights = 1.0 - (y.sum(dim=0).clamp(400, 6000) / y.sum())
            instance_weights = (y * class_weights).max(dim=1)[0]
        train_sampler = WeightedRandomSampler(instance_weights, len(X), replacement=True)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=self.batch_size)

        if self.hierarchical:
            model = BertForHierarchicalMultilabelSequenceClassification.from_pretrained(self.model_name,
                                                                                        num_labels=len(y[0]))
            model.set_hierarchy(self.label_hierarchy)
        else:
            model = BertForMultilabelSequenceClassification.from_pretrained(self.model_name,
                                                                            num_labels=len(y[0]))
        model.loss = self.loss
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
          {'params': [p for n, p in param_optimizer
                      if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in param_optimizer
                      if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        #optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        num_total_steps = self.n_epochs * (len(train_dataloader.sampler)
                                           // self.batch_size
                                           // self.gradient_accumulation_steps)
        num_warmup_steps = int(num_total_steps * 0.15)

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=3e-5, correct_bias=False)

        # PyTorch scheduler
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=num_warmup_steps,
                                         t_total=num_total_steps)

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(DEVICE)
        if self.hierarchical:
            class_weights = y.sum(dim=0)
            for lvl, idx in self.label_hierarchy.items():
                lo, hi = class_weights[idx].min(), class_weights[idx].max()
                class_weights[idx] = 1.0 - class_weights[idx].clamp(2*lo, hi*0.75) / class_weights[idx].sum()
        else:
            class_weights = (1.0 - (y.sum(dim=0).clamp(200, 4000) / y.sum()))

        class_weights = class_weights.to(DEVICE)
        i_step = 1  # stop pylint from complaining
        start_time = time.time()
        gradient_accumulation_steps = self.gradient_accumulation_steps
        epochs = trange(self.n_epochs, desc="Epoch")
        for i_epoch in epochs:
            steps = tqdm(train_dataloader,
                         total=len(train_dataloader.sampler) // train_dataloader.batch_size + 1,
                         desc='Mini-batch')
            train_loss = 0
            batch_loss = 0
            self.model = model.train()
            for i_step, batch in enumerate(steps):
                batch = (b.to(DEVICE) for b in batch)
                batch_input, batch_targets, batch_masks = batch
                loss, *_ = model(batch_input,
                                 label_hierarchy=self.label_hierarchy,
                                 labels=batch_targets,
                                 attention_mask=batch_masks,
                                 class_weights=class_weights)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                batch_loss += loss.item()
                train_loss += loss.item()
                if (gradient_accumulation_steps <= 1
                        or (i_step + 1) % gradient_accumulation_steps == 0):
                    batch_loss = batch_loss / self.gradient_accumulation_steps
                    steps.set_postfix_str(f'loss {batch_loss:.4f} || '
                                          f'avg. loss {train_loss / (i_step + 1):.4f}')

                    with open('loss.txt', 'a') as fh:
                        fh.write(f'batch\t{i_step}\t{batch_loss:.10f}\ttrain\n')
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    batch_loss = 0

            if callable(self.post_epoch_hook):
                self.post_epoch_hook(self, i_epoch, dev)

            with open('loss.txt', 'a') as fh:
                fh.write(f'epoch\t{i_epoch}\t{train_loss / i_step:.10f}\ttrain\n')
            steps.close()
            epochs.set_postfix_str(f'avg. loss {train_loss / i_step:.4f}')
        self.model = model.to('cpu')
        return self

    def predict(self, X, return_all_scores=False):
        tokens, masks = self.tokenize(X)
        tokens = torch.LongTensor(tokens)
        masks = torch.LongTensor(masks)

        data = TensorDataset(tokens, masks)
        dl = DataLoader(data,
                        sampler=SequentialSampler(data),
                        batch_size=self.batch_size)
        outputs = []
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model
        with torch.no_grad():
            model = model.eval().to(DEVICE)
            for batch in dl:
                batch = (b_.to(DEVICE) for b_ in batch)
                tokens_batch, mask_batch = batch
                output = model(tokens_batch, token_type_ids=None,
                               attention_mask=mask_batch, labels=None,
                               label_hierarchy=self.label_hierarchy)
                if return_all_scores:
                    outputs.append((output.detach(), mask_batch.detach()))
                else:
                    prob = output[0].exp().detach()
                    outputs.append(prob)
        return outputs

class BertForMultilabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_weights)
        self.loss = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, position_ids=None, head_mask=None,
                class_weights=None, **kwargs):
        outputs = self.bert(input_ids,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if not self.loss or self.loss == 'bce':
            loss_fct = BCEWithLogitsLoss(weight=class_weights)
        elif self.loss == 'multilabel-softmargin':
            loss_fct = MultiLabelSoftMarginLoss()
        else:
            raise ValueError(f'Unknown loss function {self.loss}')

        if labels is not None:
            loss = loss_fct(logits, labels.float())
            outputs = (loss,) + outputs

        return outputs


class BertForHierarchicalMultilabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropouts = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()

    def set_hierarchy(self, label_hierarchy):
        self.label_hierarchy = label_hierarchy
        num_labels = set()
        for k, v in label_hierarchy.items():
            num_labels.update(v)
            dropout = nn.Dropout(self.config.hidden_dropout_prob)
            classifier = nn.Linear(self.config.hidden_size, len(v))
            self.dropouts[f'{k}'] = dropout
            self.classifiers[f'{k}'] = classifier
        self.num_labels = len(num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, label_hierarchy=None, token_type_ids=None, attention_mask=None,
                labels=None, position_ids=None, head_mask=None,
                class_weights=None, **kwargs):
        if label_hierarchy is None:
            label_hierarchy = self.label_hierarchy
        outputs = self.bert(input_ids,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]

        loss_fct = MultiLabelSoftMarginLoss()

        loss = torch.FloatTensor([0])
        logits = torch.zeros((input_ids.size()[0], self.num_labels), dtype=torch.float32, device=input_ids.device)

        # go through the label hierarchy and get predictions from the corresponding models
        for k, idx_ in label_hierarchy.items():
            pooled_output_ = self.dropouts[f'{k}'](pooled_output)
            logits_ = self.classifiers[f'{k}'](pooled_output_)
            if labels is not None:
                labels_ = labels[:, idx_]
                loss_fct = BCEWithLogitsLoss(weight=class_weights[idx_] if class_weights is not None else None)
                loss += loss_fct(logits_, labels_.float())
            logits[:, idx_] = logits_

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            outputs = (loss,) + outputs

        return outputs
