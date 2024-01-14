import torch

def transform(x, window_size, psd_max_ori):
    # Scaling factor for transformation
    scale = 8. / 3.

    # Apply scaling to the input
    z = scale * (x / window_size)

    # Compute power spectral density (PSD)
    psd = torch.square(z)

    # Transform PSD based on max PSD value
    PSD = torch.pow(torch.tensor(10.), 9.6) / psd_max_ori * psd

    return PSD

def compute_loss_th(delta, window_size, th_batch, psd_max_batch):
    # ReLU activation function
    f = torch.nn.ReLU()

    # List to store individual losses for each example in the batch
    losses = []

    # Iterate over examples in the batch
    for i in range(delta.size(0)):
        # Transform the input using specified window size and PSD max
        logits_delta = transform(delta[i, :], window_size, psd_max_batch[i])

        # Compute threshold loss using ReLU and mean over the logits
        loss_th = f(logits_delta.squeeze() - th_batch[i]).mean()

        # Append the loss to the list
        losses.append(loss_th)

    # Stack individual losses into a tensor
    loss_th = torch.stack(losses)

    return loss_th
