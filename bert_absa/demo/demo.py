from transformers import BertTokenizer
from transformers import AdamW
from transformers import BertForSequenceClassification

import torch


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt',
                     padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']


labels = torch.tensor([1, 0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
