"""SVM classifier"""
import torch
import torch.nn as nn

# reimplement A1 code but with pytorch
class MulticlassSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MulticlassSVM, self).__init__()
        # Simple model with a linear layer (equivalent to SVM)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def hinge_loss(self, outputs, labels, margin=1.0):
        # SVM Hinge Loss
        num_samples = outputs.size(0)
        correct_class_scores = outputs[range(num_samples), labels]

        print(outputs.shape)
        print(labels.shape)
        
        # Compute the margin-based loss for all classes except the correct class
        margins = torch.max(outputs - correct_class_scores + margin, 0)
        margins[range(num_samples), labels] = 0  # Ignore the correct class
        loss = torch.sum(margins) / num_samples
        return loss