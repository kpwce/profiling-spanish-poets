import torch
import torch.nn as nn

class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Basic forward pass, just one layer"""
        return self.l1(x)

def svm_loss(outputs, labels, margin=1.0):
    """
    Simple hinge loss
    """
    batch_size = outputs.size(0)
    correct_class_scores = outputs[torch.arange(batch_size), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_class_scores + margin, min=0)
    margins[torch.arange(batch_size), labels] = 0
    return margins.sum() / batch_size