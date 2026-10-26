"""Explicit client-local optimizers; fresh state for each training phase/round."""
import math


def optimizer_settings(settings=None):
    """Missing historical settings retain the original plain SGD behavior."""
    if settings is None:
        return {'name': 'sgd', 'momentum': 0.0, 'weight_decay': 0.0}
    if not isinstance(settings, dict) or settings.get('name') not in ('sgd', 'adamw'):
        raise ValueError('optimizer name must be sgd or adamw')
    required = {'name', 'weight_decay'}
    if settings['name'] == 'sgd':
        required.add('momentum')
    if set(settings) != required:
        raise ValueError('optimizer fields must explicitly declare weight_decay and SGD momentum')
    for key in required - {'name'}:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'optimizer {key} must be finite and nonnegative')
    if settings['name'] == 'sgd' and settings['momentum'] >= 1:
        raise ValueError('optimizer momentum must be in [0,1)')
    return dict(settings)


def make_optimizer(parameters, *, learning_rate, settings=None):
    # Keep PyTorch optional for data preparation/configuration inspection.
    import torch
    options = optimizer_settings(settings)
    if options['name'] == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=options['momentum'],
                               weight_decay=options['weight_decay'])
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=options['weight_decay'],
                             betas=(0.9, 0.999), eps=1e-8)
