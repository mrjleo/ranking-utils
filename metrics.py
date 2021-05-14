import torch


def average_precision(predictions: torch.FloatTensor, labels: torch.IntTensor) -> torch.FloatTensor:
    """Compute the average precision for a single query.

    Arguments:
        predictions (torch.FloatTensor): A list of predictions
        labels (torch.IntTensor): A list of labels

    Returns:
        torch.FloatTensor: The average precision
    """
    prediction_indices = predictions.argsort(descending=True)
    label_indices, = torch.where(labels > 0)
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(prediction_indices):
        if p in label_indices and p not in prediction_indices[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return torch.as_tensor(score / max(1, len(label_indices)), device=predictions.device)


def reciprocal_rank(predictions: torch.FloatTensor, labels: torch.IntTensor) -> torch.FloatTensor:
    """Compute the reciprocal rank for a single query.

    Arguments:
        predictions (torch.FloatTensor): A list of predictions
        labels (torch.IntTensor): A list of labels

    Returns:
        torch.FloatTensor: The reciprocal rank
    """
    prediction_indices = predictions.argsort(descending=True)
    label_indices, = torch.where(labels > 0)
    for rank, item in enumerate(prediction_indices):
        if item in label_indices:
            return torch.as_tensor(1.0 / (rank + 1), device=predictions.device)
    return torch.as_tensor(0.0, device=predictions.device)
