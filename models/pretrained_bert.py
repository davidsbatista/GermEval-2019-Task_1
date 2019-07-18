import os
import time
import logging
import functools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
from tqdm import tqdm, trange

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from utils import pre_processing


class PretrainedBert:
    def __init__(self, model_name='bert-base-multilingual-cased', batch_size=8,
                 gradient_accumulation_steps=8, n_epochs=3):
        self.batch_size = batch_size
        self.label_map = {k: v for v, k in enumerate(['O', 'X', '[CLS]', '[SEP]'], 1)}
        self.model = None
        self.model_path = None
        self.tokenizer = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_epochs = n_epochs
        self.model_name = model_name

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
                                         lowercase=True,
                                         simple=False,
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

    def fit(self, X, y):
        tokens, masks = self.tokenize(X)
        tokens = torch.LongTensor(tokens)
        masks = torch.LongTensor(masks)

        y = torch.LongTensor(y)

        train_data = TensorDataset(tokens, y, masks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=self.batch_size)

        model = BertForMultilabelSequenceClassification.from_pretrained(self.model_name,
                                                                        num_labels=len(y[0]))
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
          {'params': [p for n, p in param_optimizer
                      if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in param_optimizer
                      if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # param_optimizer = list(model.classifier.named_parameters())
        # optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        num_train_optimization_steps = (len(X) // self.batch_size //
                                        self.gradient_accumulation_steps) * self.n_epochs

        num_warmup_steps = 100
        num_total_steps = 1000

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=3e-5, correct_bias=False)

        # PyTorch scheduler
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=num_warmup_steps,
                                         t_total=num_total_steps)

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(DEVICE)
        i_step = 1  # stop pylint from complaining
        start_time = time.time()
        gradient_accumulation_steps = self.gradient_accumulation_steps
        for i_epoch in trange(self.n_epochs, desc="Epoch"):
            steps = tqdm(train_dataloader,
                         total=len(tokens) // train_dataloader.batch_size + 1,
                         desc='Mini-batch')
            train_loss = 0
            model.train()
            for i_step, batch in enumerate(steps):
                batch = (b.to(DEVICE) for b in batch)
                batch_input, batch_targets, batch_masks = batch
                loss, *_ = model.forward(batch_input, labels=batch_targets,
                                        attention_mask=batch_masks)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                train_loss += loss.item()
                steps.set_postfix_str(f'avg. loss {train_loss / (i_step + 1):.4f}')
                if (gradient_accumulation_steps <= 1
                        or (i_step + 1) % gradient_accumulation_steps == 0):
                    optimizer.step()
                    optimizer.zero_grad()
            model.evaluate()
            steps.close()
        model = model.to('cpu')
        return model

    def predict(self, X):
        pass


class BertForMultilabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultilabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            outputs = (loss,) + outputs

        return outputs