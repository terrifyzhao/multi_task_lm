import torch
from transformers.models.bert.modeling_bert import BertModel, BertConfig
from transformers import BertTokenizer
from annlp import ptm_path

path = ptm_path('roberta')
tokenizer = BertTokenizer.from_pretrained(path)
model = BertModel.from_pretrained(path)
embedding_weight = model.embeddings.word_embeddings.weight[0:13317]

print(123)

path = ptm_path('roberta_embed')
config = BertConfig.from_pretrained(path)
model2 = BertModel(config)
# model.embeddings.word_embeddings.weight = embedding_weight

i = 0
for p1, p2, in zip(model.parameters(), model2.parameters()):
    i += 1
    if i == 1:
        p2.data.copy_(embedding_weight)
    else:
        p2.data.copy_(p1.data)

i = 0
for p1, p2, in zip(model.parameters(), model2.parameters()):
    i += 1
    if i > 1:
        print(p1.equal(p2))
        # if p1 != p2:
        #     print(1)

torch.save(model2.state_dict(), 'pytorch_model.bin')
