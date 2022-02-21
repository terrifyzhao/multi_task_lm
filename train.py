import torch
from annlp import *
# from model import RoFormerForMultiTask
from model2 import BertForMultiTask
from transformers import BertTokenizer
from dataloader import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

fix_seed()
path = ptm_path('roberta')
BATCH_SIZE = 8
EPOCHS = 20
MAX_LENGTH = 32
device = get_device()
tokenizer = BertTokenizer.from_pretrained(path)


def read_data(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    text = df['text'].tolist()
    label_unique = df['label'].unique()
    label_dict = {label_unique[i]: i for i in range(len(label_unique))}
    label = [label_dict[l] for l in df['label'].tolist()]

    train_text, dev_text, train_label, dev_label = train_test_split(text, label, test_size=test_size,
                                                                    random_state=random_state)

    train_encoding = tokenizer(text=train_text,
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=MAX_LENGTH)
    train_dataset = BaseDataset(train_encoding, train_label)

    dev_encoding = tokenizer(text=dev_text,
                             return_tensors='pt',
                             truncation=True,
                             padding=True,
                             max_length=MAX_LENGTH)
    dev_dataset = BaseDataset(dev_encoding, dev_label)

    return train_dataset, dev_dataset, len(label_unique)


news_10_dataset_train, news_10_dataset_dev, news_10_label_len = read_data('data/news_10.csv')
sentiment_hotel_dataset_train, sentiment_hotel_dataset_dev, sentiment_hotel_label_len = read_data(
    'data/sentiment_hotel.csv')
train_data_loader = MultiTaskDataLoader([news_10_dataset_train, sentiment_hotel_dataset_train], BATCH_SIZE)

news_10_dataloader_dev = DataLoader(news_10_dataset_dev, BATCH_SIZE)
sentiment_hotel_dataloader_dev = DataLoader(sentiment_hotel_dataset_dev, BATCH_SIZE)
dev_data_loader = [news_10_dataloader_dev, sentiment_hotel_dataloader_dev]

model = BertForMultiTask.from_pretrained(path, num_labels_list=[news_10_label_len, sentiment_hotel_label_len])
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)
model.train()

handlers = []

pbar = tqdm(range(10000))
for step in pbar:
    batch = next(train_data_loader)
    all_loss = 0
    optim.zero_grad()
    for index in range(len(batch)):
        output = model(batch[index]['input_ids'].to(device),
                       batch[index]['attention_mask'].to(device),
                       labels=batch[index]['labels'].to(device),
                       task_id=index)
        loss = output.loss
        all_loss += loss.item()

        # hook = loss.register_hook(lambda grad: grad / torch.norm(grad))
        # model.parameters()
        # loss = loss / 2【
        loss.backward()

        from torch import tensor

        for params in model.parameters():
            if params.grad is not None:
                params.grad = params.grad / torch.norm(params.grad)
        # hook.remove()
    optim.step()

    all_loss = all_loss / len(dev_data_loader)
    pbar.update()
    pbar.set_description(f'loss:{all_loss:.4f}')

    # 100步做一下验证
    if step >= 100 and step % 100 == 0:
        with torch.no_grad():
            model.eval()
            # acc = 0
            for i, loader in enumerate(dev_data_loader):
                all_pred = []
                all_label = []
                for batch in loader:
                    output = model(batch['input_ids'].to(device),
                                   batch['attention_mask'].to(device),
                                   task_id=i)
                    all_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
                    all_label.extend(batch['labels'].numpy())
                acc = accuracy_score(all_label, all_pred)
                print(f'acc {i + 1}:{acc:.4f}')
            # acc = acc / len(dev_data_loader)
            # print('dev acc:', acc)
        model.train()

for h in handlers:
    h.remove()
