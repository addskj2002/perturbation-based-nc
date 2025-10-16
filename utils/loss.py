
import torch
from tqdm import tqdm


def get_simclr_loss(model, data, device, tau=0.07):

    left = model(data[:, 0])
    right = model(data[:, 1])
    left = torch.nn.functional.normalize(left, dim=1)
    right = torch.nn.functional.normalize(right, dim=1)
    transformed_data = torch.stack([left, right], dim=1)

    batch_size = transformed_data.shape[0]
    contrast_count = transformed_data.shape[1]
    mask = torch.eye(batch_size, dtype=torch.float32)

    contrast_feature = torch.cat(torch.unbind(transformed_data, dim=1), dim=0)

    # compute logits
    anchor_dot_contrast = contrast_feature @ contrast_feature.T / tau
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(contrast_count, contrast_count).to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    return -mean_log_prob_pos.flatten()


def get_byol_loss(model, data, device):
    online_result = model(data)
    target_result = model(data, is_target=True)
    return 1 - torch.nn.CosineSimilarity(dim=1)(online_result, target_result)


def get_moco_loss(model, data, device):
    return model.contrastive_loss(data[:, 0], data[:, 1])[0]


def get_simclr_loss_diff(model1, model2, data, device, tau=0.07):
    return (
        get_simclr_loss(model1, data, device, tau) -
        get_simclr_loss(model2, data, device, tau)
    )

def get_byol_loss_diff(model1, model2, data, device):
    return (
        get_byol_loss(model1, data, device) -
        get_byol_loss(model2, data, device)
    )

def get_moco_loss_diff(model1, model2, data, device):
    return (
        get_moco_loss(model1, data, device) -
        get_moco_loss(model2, data, device)
    )