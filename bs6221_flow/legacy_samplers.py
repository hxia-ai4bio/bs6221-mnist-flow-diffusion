import torch
from .models import cosine_schedule
DIFFUSION_STEPS = 100
schedule = cosine_schedule(100)

def validate_labels(labels, batch_size):
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise TypeError('labels 必须是 torch.long 整数 tensor。')
    if labels.shape != (batch_size,) or batch_size < 1:
        raise ValueError('每张图片需要一个标签，batch 不能为空。')
    if not ((labels >= 0) & (labels <= 9)).all():
        raise ValueError('数字标签必须在 0–9 之间。')
    return labels

@torch.no_grad()
def sample_flow(model, initial_noise, labels, steps=80):
    model.eval()
    DEVICE = next(model.parameters()).device
    state = initial_noise.to(DEVICE).clone()
    labels = validate_labels(labels, len(state)).to(DEVICE)
    snapshots = [state.cpu().clone()]
    keep = {round(i * steps / 5) for i in range(1, 6)}
    h = 1 / steps
    for j in range(steps):
        t = torch.full((len(state),), j / steps, device=DEVICE)
        velocity = model(state, t, labels)
        proposal = state + h * velocity
        next_t = torch.full_like(t, (j + 1) / steps)
        state = state + 0.5 * h * (velocity + model(proposal, next_t, labels))
        if not torch.isfinite(state).all():
            raise RuntimeError('Flow 生成出现非有限状态。')
        if j + 1 in keep:
            snapshots.append(state.cpu().clone())
    return (state.cpu(), torch.stack(snapshots))

@torch.no_grad()
def sample_diffusion(model, initial_noise, labels, seed=2027, noise_group_size=None):
    model.eval()
    DEVICE = next(model.parameters()).device
    schedule = {k: v.to(DEVICE) for k,v in cosine_schedule(DIFFUSION_STEPS).items()}
    generator = torch.Generator().manual_seed(seed)
    state = initial_noise.to(DEVICE).clone()
    labels = validate_labels(labels, len(state)).to(DEVICE)
    if noise_group_size is not None and (noise_group_size < 1 or len(state) % noise_group_size):
        raise ValueError('噪声分组大小必须为正数且整除 batch。')
    snapshots = [state.cpu().clone()]
    remaining = {80, 60, 40, 20, 0}
    for j in reversed(range(DIFFUSION_STEPS)):
        times = torch.full((len(state),), j / (DIFFUSION_STEPS - 1), device=DEVICE)
        predicted_noise = model(state, times, labels)
        a = schedule['abar'][j]
        predicted_clean = ((state - (1 - a).sqrt() * predicted_noise) / a.sqrt()).clamp(-1, 1)
        mean = schedule['coef_x0'][j] * predicted_clean + schedule['coef_xt'][j] * state
        if j > 0:
            shape = (noise_group_size or len(state), *state.shape[1:])
            noise = torch.randn(shape, generator=generator)
            if noise_group_size is not None:
                noise = noise.repeat(len(state) // noise_group_size, 1, 1, 1)
            noise = noise.to(DEVICE)
            state = mean + schedule['posterior_var'][j].sqrt() * noise
        else:
            state = mean
        if not torch.isfinite(state).all():
            raise RuntimeError('Diffusion 生成出现非有限状态。')
        if j in remaining:
            snapshots.append(state.cpu().clone())
    return (state.cpu(), torch.stack(snapshots))
