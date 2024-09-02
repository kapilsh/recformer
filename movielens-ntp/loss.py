import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class DedupCrossEntropyLoss(nn.Module):
    """
    Custom loss function that combines cross-entropy loss with a penalty term
    for tokens that are repeated from the input sequence.
    """
    def __init__(self, penalty_weight = 1.0):
        super(DedupCrossEntropyLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.ce_loss = nn.CrossEntropyLoss()
        logger.info(f"Penalty weight: {penalty_weight}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, input_tokens: torch.Tensor):
        # Calculate the standard cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        if self.penalty_weight == 0:
            return ce_loss
        token_output = torch.argmax(logits, dim=1)
        duplicated_masks = torch.eq(input_tokens, token_output.unsqueeze(-1)).any(dim=-1).float()
        penalty = duplicated_masks * self.penalty_weight
        loss = ce_loss + penalty.mean()
        return loss