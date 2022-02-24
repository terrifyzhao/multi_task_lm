import numpy as np
from annlp import fix_seed, ptm_path, get_device, print_sentence_length
# from model import RoFormerForMultiTask
import pandas as pd
from optimizer import AdamW, Adam
from model2 import BertForMultiTask
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from dataloader import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
from train_step import grad_accumulation_step
import os
import pickle

torch.cuda.is_available()

fix_seed()
path = ptm_path('roberta')
BATCH_SIZE = 128
EPOCHS = 20
device = get_device()
tokenizer = BertTokenizer.from_pretrained(path)


def read_data(path, max_length, test_size=0.1, random_state=42):
    df = pd.read_csv(path)
    df = df[0:20000]
    text = df['text'].tolist()
    # print(len(text))
    # print_sentence_length(text)
    # return None,None,1
    label_unique = df['label'].unique()
    label_dict = {label_unique[i]: i for i in range(len(label_unique))}
    label = [label_dict[l] for l in df['label'].tolist()]

    train_text, dev_text, train_label, dev_label = train_test_split(text, label, test_size=test_size,
                                                                    random_state=random_state)

    train_encoding = tokenizer(text=train_text,
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=max_length)
    train_dataset = BaseDataset(train_encoding, train_label)

    dev_encoding = tokenizer(text=dev_text,
                             return_tensors='pt',
                             truncation=True,
                             padding=True,
                             max_length=max_length)
    dev_dataset = BaseDataset(dev_encoding, dev_label)

    return train_dataset, dev_dataset, len(label_unique)


if os.path.exists('data/data.bin'):
    train_data_loader, dev_data_loader, num_labels_list = pickle.load(open('data/data.bin', 'rb'))
else:
    data_paths = ['fudan_news', 'sentiment', 'thucnews', 'tnews', 'weibo',
                  'tc_impact', 'tc_opinion', 'tc_push', 'tc_reminder', 'tc_sentiment']
    max_len_list = [32, 256, 32, 32, 128, 128, 128, 128, 128, 128]
    batch_size_list = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    train_datasets = []
    dev_data_loader = []
    num_labels_list = []
    for p, l in zip(data_paths, max_len_list):
        data_path = 'data/' + p + '.csv'
        print('processing {}'.format(data_path))
        train, dev, label_len = read_data(data_path, max_length=l)
        train_datasets.append(train)
        dev_data_loader.append(DataLoader(dev, batch_size=BATCH_SIZE))
        num_labels_list.append(label_len)
    train_data_loader = MultiTaskDataLoader(train_datasets, batch_size_list)
    pickle.dump([train_data_loader, dev_data_loader, num_labels_list], open('data/data.bin', 'wb'))

model = BertForMultiTask.from_pretrained(path, num_labels_list=num_labels_list)
# model.load_state_dict(torch.load('best_model.pth', map_location=device))
# torch.save(model.bert.state_dict(), 'pytorch_model.bin')
# exit(0)

total_step = 2000
optim = AdamW(model.parameters(), lr=5e-5)
warm_schedule = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=int(total_step * 0.1),
                                                num_training_steps=total_step)

model.to(device)
model.train()

amp = None
if torch.cuda.is_available():
    amp = __import__('apex').amp
    model, optim = amp.initialize(model, optim, opt_level="O1")

best_acc = 0
pbar = tqdm(range(total_step))
for step in pbar:
    batch = next(train_data_loader)
    all_loss = grad_accumulation_step(batch, optim, model, amp)
    warm_schedule.step()
    pbar.update()
    pbar.set_description(f'loss:{all_loss:.4f}')

    # 100步做一下验证
    if step >= 500 and step % 100 == 0:
        model.eval()
        acc_list = []
        for i, loader in enumerate(dev_data_loader):
            all_pred = []
            all_label = []
            for batch in loader:
                with torch.no_grad():
                    output = model(batch['input_ids'].to(device),
                                   batch['attention_mask'].to(device),
                                   task_id=i)
                all_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
                all_label.extend(batch['labels'].numpy())
            acc = f1_score(all_label, all_pred, average='macro')
            acc_list.append(acc)
        print(f'dev f1: {np.mean(acc_list):.4f}')
        for i, acc in enumerate(acc_list):
            print(f'{i}-f1: {acc:.4f}')

        cur_acc = sum(acc_list) / len(acc_list)
        if cur_acc > best_acc:
            best_acc = cur_acc
            print('save model')
            torch.save(model.bert.state_dict(), 'pytorch_model.bin')

        print('best f1:', best_acc)
        print('-' * 50)
        model.train()


