#!/usr/bin/env python
from typing import Dict
import sys
import os
import numpy as np
import pickle
from datetime import datetime
from random import shuffle
import torch
import torch.cuda
import torch.nn
import copy
import time

import random

from config import config
from model.sequence_labeling import BiRecurrentConvCRF4NestedNER
from training.logger import get_logger
from training.utils import adjust_learning_rate, clip_model_grad, create_opt
from training.utils import pack_target, unpack_prediction
from util.evaluate import evaluate, evaluate_raw, evaluate_raw_tagged
from util.utils import Alphabet, load_dynamic_config

entity_tags = ['ACTIVITY', 'ADMINISTRATION_ROUTE', 'AGE', 'ANATOMY', 'CHEM', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 
            'DISO', 'FACILITY', 'FINDING', 'FOOD', 'GENE', 'HEALTH_CARE_ACTIVITY', 'INJURY_POISONING', 'LABPROC', 
            'LIVB', 'LOCATION', 'MEDPROC', 'MENTALPROC', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PERCENT', 'PERSON', 
            'PHYS', 'PRODUCT', 'PROFESSION', 'SCIPROC', 'STATE_OR_PROVINCE', 'TIME']

def get_f1(model: BiRecurrentConvCRF4NestedNER, mode: str, file_path: str = None) -> float:
    with torch.no_grad():
        model.eval()

        pred_all, pred, recall_all, recall = 0, 0, 0, 0
        gold_cross_num = 0
        pred_cross_num = 0
        if mode == 'dev':
            batch_zip = zip(dev_input_ids_batches,
                            dev_input_mask_batches,
                            dev_first_subtokens_batches,
                            dev_last_subtokens_batches,
                            dev_label_batches,
                            dev_mask_batches)
        elif mode == 'test':
            batch_zip = zip(test_input_ids_batches,
                            test_input_mask_batches,
                            test_first_subtokens_batches,
                            test_last_subtokens_batches,
                            test_label_batches,
                            test_mask_batches)
        else:
            raise ValueError

        f = None
        if file_path is not None:
            f = open(file_path, 'w')

        import copy
        score_dict = copy.deepcopy({entity_tag : copy.deepcopy({"tp" : 0, "fp" : 0, "fn" : 0}) for entity_tag in entity_tags})
        atp, afp, afn = 0, 0, 0

        for input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch, label_batch, mask_batch \
                in batch_zip:
            input_ids_batch_var = torch.LongTensor(np.array(input_ids_batch))
            input_mask_batch_var = torch.LongTensor(np.array(input_mask_batch))
            mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8))
            # if config.if_gpu:
            #    input_ids_batch_var = input_ids_batch_var.cuda()
            #    input_mask_batch_var = input_mask_batch_var.cuda()
            #    mask_batch_var = mask_batch_var.cuda()

            pred_sequence_entities = model.predict(input_ids_batch_var,
                                                   input_mask_batch_var,
                                                   first_subtokens_batch,
                                                   last_subtokens_batch,
                                                   mask_batch_var)
            pred_entities = unpack_prediction(model, pred_sequence_entities)
            rtp, rfp, rfn = evaluate_raw(label_batch, pred_entities)
            atp += rtp
            afp += rfp
            afn += rfn
            batch_counts = copy.deepcopy(evaluate_raw_tagged(label_batch, pred_entities, Alphabet(entity_tags, 0)))

            for tag in entity_tags:
                print(batch_counts[tag])
                score_dict[tag]["tp"] += batch_counts[tag]["tp"]
                score_dict[tag]["fp"] += batch_counts[tag]["fp"]
                score_dict[tag]["fn"] += batch_counts[tag]["fn"]

            if file_path is not None:
                for input_ids, input_mask, first_subtokens, last_subtokens, mask, label, preds \
                        in zip(input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch,
                               mask_batch, label_batch, pred_entities):
                    words = []
                    for t, m in zip(input_ids, input_mask):
                        if m == 0:
                            break
                        words.append(voc_dict.get_instance(t))
                    f.write(' '.join(words) + '\n')

                    labels = []
                    for l in sorted(label, key=lambda x: (x[0], x[1], x[2])):
                        s = first_subtokens[l[0]]
                        e = last_subtokens[l[1] - 1]
                        labels.append("{},{} {}".format(s, e, label_dict.get_instance(l[2])))
                    f.write('|'.join(labels) + '\n')

                    labels = []
                    for p in sorted(preds, key=lambda x: (x[0], x[1], x[2])):
                        s = first_subtokens[p[0]]
                        e = last_subtokens[p[1] - 1]
                        labels.append("{},{} {}".format(s, e, label_dict.get_instance(p[2])))
                    f.write('|'.join(labels) + '\n')

                    f.write('\n')

        if file_path is not None:
            f.close()

        miprec = atp / (atp + afp + 1e-10)
        mirec = atp / (atp + afn + 1e-10)
        mif1 = (2 * miprec * mirec) / (miprec + mirec + 1e-10)
        logger.info("{} micro precision: {:.2f}%, micro recall: {:.2f}%, micro F1: {:.2f}%"
                    .format(mode, miprec * 100., mirec * 100., mif1 * 100.))

        precs = []
        recs = []
        f1s = []
        print(score_dict)
        for tag in entity_tags:
            tp = score_dict[tag]["tp"]
            fp = score_dict[tag]["fp"]
            fn = score_dict[tag]["fn"]
            precs.append(tp / (tp + fp + 1e-10))
            recs.append(tp / (tp + fn + 1e-10))
        maprec = sum(precs) / len(precs)
        marec = sum(recs) / len(recs)
        maf1 = 2 * maprec * marec / (maprec + marec + 1e-10)
        logger.info("{} macro precision: {:.2f}%, macro recall: {:.2f}%, macro F1: {:.2f}%"
                    .format(mode, maprec * 100., marec * 100., maf1 * 100.))

        return mif1

f = open(config.test_data_path, 'rb')
test_filename_batches, \
    test_sent_id_batches, \
    test_input_ids_batches, \
    test_input_mask_batches, \
    test_first_subtokens_batches, \
    test_last_subtokens_batches, \
    test_label_batches, \
    test_mask_batches \
    = pickle.load(f)
f.close()

log_file_path = 'nerelbio_model_220729_030658.tmp'
this_model_path = "second-best/dumps/nerelbio_model_220729_030658.pt"
misc_config: Dict[str, Alphabet] = pickle.load(open(config.config_data_path, 'rb'))
voc_dict, label_dict = load_dynamic_config(misc_config)
logger = get_logger('Nested Mention Test', file=log_file_path)

best_model = BiRecurrentConvCRF4NestedNER(config.bert_model, label_dict,
                                         hidden_size=config.hidden_size, layers=config.layers,
                                         lstm_dropout=config.lstm_dropout)
best_model.load_state_dict(torch.load(this_model_path, map_location = 'cpu'))
best_model.eval()

cur_time = time.time()
f1 = get_f1(best_model, 'test', file_path=this_model_path[:-3] + '.result.txt')

os.rename(log_file_path, this_model_path[:-3] + '.log.txt')