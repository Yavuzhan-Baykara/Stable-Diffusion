import torch
from tqdm.notebook import tqdm

num_steps = 500

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=num_steps,
                           device="cuda",
                           eps=1e-3, y=None):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x


def PC_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               x_shape=(1, 28, 28),
               num_steps=num_steps,
               device="cuda",
               eps=1e-3, y=None):

    def predictor_step(x, t, step_size):
        g = diffusion_coeff(t)
        mean_x = x + (g**2)[:, None, None, None] * score_model(x, t, y=y) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
        return x

    def corrector_step(x, t, step_size, num_corrector_steps=1):
        for _ in range(num_corrector_steps):
            grad = score_model(x, t, y=y)
            noise = torch.randn_like(x)
            x = x + step_size * grad + torch.sqrt(2 * step_size) * noise
        return x

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x = predictor_step(x, batch_time_step, step_size)
            x = corrector_step(x, batch_time_step, step_size)
    return x

