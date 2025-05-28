import torch.nn as nn
from transformers import BertModel

class BETOClassifier(nn.Module):
    def __init__(self, bert_model_name='dccuchile/bert-base-spanish-wwm-cased', num_classes, dropout_prob=0.3):
        super(BETOClassifier, self).__init__()
        
        self.beto = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.beto(input_ids=input_ids, attention_mask=attention_mask)
        class_out = outputs.pooler_output # token embedding
        dropout_out = self.dropout(class_out)
        out = self.classifier(dropout_out)
        return out
