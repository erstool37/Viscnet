def as_batch_vector(tensor, *, dtype=None, device=None):
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor.view(-1)
