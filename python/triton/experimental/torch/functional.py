"""Private functionalization copies retaining shared Tensor storage."""


def clone_arguments(arguments, written):
    import torch
    groups = {}
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            groups.setdefault(value.untyped_storage()._cdata, []).append((name, value))
    result = dict(arguments)
    for members in groups.values():
        if not any(name in written for name, _ in members):
            continue
        prototype = members[0][1]
        storage = prototype.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8, device=prototype.device).set_(
            storage, 0, (storage.nbytes(),), (1,))
        private = raw.clone()
        for name, value in members:
            typed = private[:storage.nbytes() // value.element_size() * value.element_size()].view(value.dtype)
            result[name] = typed.as_strided(value.shape, value.stride(), value.storage_offset())
    return result
