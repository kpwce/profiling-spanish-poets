"""SVM classifier"""
import torch
import torch.nn as nn

# reimplement A1 code but with pytorch
class MulticlassSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MulticlassSVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    # optional class weight param for weighted SVM
    def hinge_loss(self, outputs, labels, margin=1.0, class_weights=None):
        num_samples = outputs.size(0)
        correct_class_scores = outputs[range(num_samples), labels]  # shape: (num_samples,)
        margins = torch.clamp(outputs - correct_class_scores.unsqueeze(1) + margin, min=0)

        margins[range(num_samples), labels] = 0
        if class_weights is not None:
            # weight each sample's margin loss
            sample_weights = class_weights[labels]
            loss = (margins.sum(dim=1) * sample_weights).mean()
        else:
            loss = margins.sum() / num_samples

        return loss
