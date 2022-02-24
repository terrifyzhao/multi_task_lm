import os
import torch
import pandas as pd
from transformers import AdamW
from annlp import fix_seed, ptm_path, get_device, Trainer, BertForMultiClassification, print_sentence_length
from sklearn.model_selection import train_test_split


def read_data(path, test_size=0.1, random_state=42):
    df = pd.read_csv(path)

    text = df['text'].tolist()
    print_sentence_length(text)
    label_unique = df['label'].unique()
    label_dict = {label_unique[i]: i for i in range(len(label_unique))}
    label = [label_dict[l] for l in df['label'].tolist()]

    return train_test_split(text, label, test_size=test_size, random_state=random_state), len(label_unique)


# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/news_10.csv')
(train_text, dev_text, train_label, dev_label), num_labels = read_data('data/tc_opinion.csv')


# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/sentiment_hotel.csv')
# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/fudan_news.csv')
# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/iflytek.csv')
# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/sentiment.csv')
# (train_text, dev_text, train_label, dev_label), num_labels = read_data('data/weibo.csv')


class MyTrainer(Trainer):

    def get_train_data(self):
        return self.tokenizer_(train_text), train_label

    def get_dev_data(self):
        return self.tokenizer_(dev_text), dev_label

    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data, mode):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        labels = data['labels'].to(self.device).long()

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        output = outputs.logits
        return outputs.loss, output, labels.cpu().numpy()

    def predict_step(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        output = outputs.logits.argmax(dim=-1).cpu().numpy()
        return output


def main(mode, seed):
    if mode == 'train':
        do_train = True
        do_dev = True
        do_test = False
        load_model = False
    else:
        do_train = False
        do_dev = False
        do_test = True
        load_model = True

    fix_seed(seed)

    max_length = 128
    batch_size = 32
    lr = 5e-5

    model_name = 'best_model.p'
    model_path = ptm_path('roberta')
    print(model_path)

    model = BertForMultiClassification.from_pretrained(model_path, num_labels=num_labels, loss=None)

    if os.path.exists(model_name) and load_model:
        print('************load model************')
        model.load_state_dict(torch.load(model_name, map_location=get_device()))

    trainer = MyTrainer(model, batch_size=batch_size, lr=lr, max_length=max_length, model_path=model_path,
                        do_train=do_train, do_dev=do_dev, do_test=do_test, test_with_label=False,
                        save_model_name=model_name, attack=False, monitor='f1', epochs=5,
                        save_metric='all_data', mix=None, augmentation=False)

    trainer.configure_metrics(do_acc=True, do_f1=True, do_recall=True, do_precision=True, do_kappa=True,
                              print_report=False, average='macro')
    trainer.run()


if __name__ == '__main__':
    import sys

    for seed in [100, 101, 102, 103, 104]:
        print('seed:', seed)
        main(sys.argv[1], seed=seed)
        print('*' * 100)
