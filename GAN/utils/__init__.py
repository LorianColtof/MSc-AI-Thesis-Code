import torch


def gradient_penalty(discriminator, samples_real, samples_generated,
                     lambda_reg=5):
    device = samples_real.device

    batch_size = samples_real.shape[0]
    alpha = torch.rand(batch_size, 1, device=device) \
        .expand(samples_real.reshape(batch_size, -1).shape) \
        .reshape(samples_real.shape)

    interpolates: torch.Tensor = alpha * samples_real + \
        ((1 - alpha) * samples_generated[:batch_size])

    interpolates_var = torch.autograd.Variable(
        interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates_var)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates_var,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    penalty = (((gradients.norm(2, dim=1) - 1) ** 2).mean()
               * lambda_reg)
    return penalty
