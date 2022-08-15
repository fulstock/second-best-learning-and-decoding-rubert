#!/usr/bin/env python
from typing import Tuple, List
from collections import defaultdict
import pickle

from config import config
from reader.reader import Reader
from util.utils import save_dynamic_config


if __name__ == "__main__":
    config.data_set = "nerelbio"
    reader = Reader(config.bert_model)
    reader.read_all_data("data/nerelbio/", "nerelbio.train", "nerelbio.dev", "nerelbio.test", utf = True)

    # print reader.train_sents[0]
    train_batches, dev_batches, test_batches = reader.to_batch(config.batch_size)
    f = open(config.train_data_path, 'wb')
    pickle.dump(train_batches, f)
    f.close()

    f = open(config.dev_data_path, 'wb')
    pickle.dump(dev_batches, f)
    f.close()

    f = open(config.test_data_path, 'wb')
    pickle.dump(test_batches, f)
    f.close()

    # misc config
    misc_dict = save_dynamic_config(reader)
    f = open(config.config_data_path, 'wb')
    pickle.dump(misc_dict, f)
    f.close()

    print("Remember to scp word vectors as well")
