import torch.nn as nn
from transformers import BertModel

class BETOClassifier(nn.Module):
    def __init__(self, num_classes, class_weights, dropout_prob=0.3):
        super(BETOClassifier, self).__init__()
        
        self.beto = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.beto.config.hidden_size, num_classes)
        self.class_weights = class_weights
        self.loss_f = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.beto(input_ids=input_ids, attention_mask=attention_mask)
        class_out = outputs.pooler_output # token embedding
        dropout_out = self.dropout(class_out)
        out = self.classifier(dropout_out)
        if labels is not None:
            loss = self.loss_f(out, labels)
        else:
            loss = None
        return {'logits': out, 'loss': loss}
