import torch

def euler_adjoint(func, y, t, grad_output):
    adj_y = grad_output[-1]
    adj_params = [torch.zeros_like(p) for p in func.parameters()]
    for i in range(len(t) - 1, 0, -1):
        dt = t[i] - t[i - 1]
        y_i = y[i - 1].detach().requires_grad_(True)
        t_i = t[i - 1]
        with torch.enable_grad():
            f_i = func(t_i, y_i)
            loss = torch.sum(f_i * adj_y * dt)
            loss.backward(retain_graph=True)
            adj_y = adj_y + y_i.grad
            y_i.grad = None
            for idx, p in enumerate(func.parameters()):
                if p.requires_grad:
                    if p.grad is not None:
                        adj_params[idx] += p.grad
                        p.grad = None
    adj_y0 = adj_y
    return adj_y0, adj_params


def euler_forward(func, y0, t):
    y = [y0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y_i = y[-1]
        f_i = func(t[i], y_i)
        y_next = y_i + dt * f_i
        y.append(y_next)
    y = torch.stack(y)  # Shape: [time_steps, batch_size, dim]
    return y


class EulerAdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, y0, t):
        y = euler_forward(func, y0, t)
        ctx.save_for_backward(t, y)
        ctx.func = func
        return y

    @staticmethod
    def backward(ctx, grad_output):
        t, y = ctx.saved_tensors
        func = ctx.func
        adj_y0, adj_params = euler_adjoint(func, y, t, grad_output)
        
        # Assign gradients to the function parameters
        for p, adj_p in zip(func.parameters(), adj_params):
            if p.requires_grad:
                if p.grad is None:
                    p.grad = adj_p.clone()
                else:
                    p.grad += adj_p.clone()
        
        # No gradient for 'func' itself (the ODE function is treated as a fixed module)
        # Gradient for 'y0'
        return (None, adj_y0, None)
    
def odeint_euler_adjoint(func, y0, t):
    y = EulerAdjointMethod.apply(func, y0, t)
    return y

