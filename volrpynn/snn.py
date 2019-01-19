import torch
import collections

LIFParameters = collections.namedtuple('LIFParameters', [
    'v_th',
    'v_rest',
    'v_reset',
    't_ref',
    'tau'
])

AdaptiveLIFParameters = collections.namedtuple('AdaptiveLIFParameters', [
    'v_th',
    'v_rest',
    'v_reset',
    't_ref',
    'tau',
    'tau_beta',
    'delta_beta'
])

default_lif_parameters = LIFParameters(
    v_th=75, # mV
    v_rest=50, # mV
    v_reset=30, # mV
    t_ref=0.001, # s
    tau=2
)

default_adaptive_lif_parameters = AdaptiveLIFParameters(
    v_th=75, # mV
    v_rest=50, # mV
    v_reset=30, # mV
    t_ref=0.001, # s
    tau=2,
    tau_beta=2,
    delta_beta=10 # mV
)

def constant_lif_parameters(param, size):
    return LIFParameters(
        v_th = torch.ones(size) * param.v_th,
        v_rest = torch.ones(size) * param.v_rest,
        v_reset = torch.ones(size) * param.v_reset,
        t_ref = torch.ones(size) * param.t_ref,
        tau = torch.ones(size) * param.tau
    )

def constant_adaptive_lif_parameters(param, size):
    return AdaptiveLIFParameters(
        v_th = torch.ones(size) * param.v_th,
        v_rest = torch.ones(size) * param.v_rest,
        v_reset = torch.ones(size) * param.v_reset,
        t_ref = torch.ones(size) * param.t_ref,
        tau = torch.ones(size) * param.tau,
        tau_beta = torch.ones(size) * param.tau_beta,
        delta_beta = torch.ones(size) * param.delta_beta
    )

class HeavySide(torch.autograd.function.Function):
    """HeavySide

    Implements a heavyside step function with an approximate
    derivative.
    """

    @staticmethod
    def forward(ctx, x, h):
        ctx.save_for_backward(x,h)
        return torch.where(x >= 0, torch.ones_like(x), torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad):
        x,h = ctx.saved_tensors
        x_grad = h_grad = None

        x_ = 1/(h*h) * (torch.where((-h <= x) * (x < 0), torch.ones_like(x), torch.zeros_like(x))
                        + torch.where((0 <= x) * (x <= h), -torch.ones_like(x), torch.zeros_like(x)))

        x_grad = x_ * grad
        return x_grad, h_grad

heavy_side = HeavySide.apply

def lif_step(v, x, refrac_count, parameters, dt):
    """Computes an euler integration update step of a leaky integrate and fire
    neuron.
    """
    # threshhold voltage, rest voltage, reset voltage, refractory time, decay constant
    v_th, v_rest, v_reset, t_ref, tau = parameters
    v -= dt * tau * (v - v_rest)

    v += torch.where(refrac_count == 0, x, torch.zeros_like(x))
    refrac_count = torch.where(refrac_count > 0, refrac_count - dt, torch.zeros_like(refrac_count))

    spike = heavy_side(v - v_th, torch.tensor([5.0])) # TODO: Figure out good value h

    refrac_count.masked_scatter_(spike.byte(), t_ref.masked_select(spike.byte()))
    v.masked_scatter_(spike.byte(), v_reset.masked_select(spike.byte()))

    return v, spike, refrac_count

def adaptive_lif_step(v, beta, x, refrac_count, parameters, dt):
    """Computes an euler integration update step of an adaptive lif
    neuron equation.
    """
    v_th, v_rest, v_reset, t_ref, tau, tau_beta, delta_beta = parameters

    v -= dt * tau * (v - v_rest)
    beta -= dt * tau_beta * beta

    v += torch.where(refrac_count == 0, x, torch.zeros_like(x))
    refrac_count = torch.where(refrac_count > 0, refrac_count - dt, torch.zeros_like(refrac_count))
    spike = heavy_side(v - (v_th + beta), torch.tensor([5.0])) # TODO: Figure out a good value for h

    refrac_count.masked_scatter_(spike.byte(), t_ref.masked_select(spike.byte()))
    v.masked_scatter_(spike.byte(), v_reset.masked_select(spike.byte()))
    beta += delta_beta * spike

    return v, spike, refrac_count, beta

def lif_euler_integrate(x, weight, lif_parameters, num_timesteps, dt):
    s = torch.zeros(num_timesteps, weight.shape[0])
    v = torch.zeros(num_timesteps, weight.shape[0])
    v[0] = lif_parameters.v_rest
    refrac_count = torch.zeros(weight.shape[0])

    lif_input = x.mm(weight.t())

    for i in range(1,num_timesteps):
        v[i], s[i], refrac_count = lif_step(v[i-1], lif_input[i], refrac_count, lif_parameters, dt)

    return v,s

class LIFTransferFunction(torch.autograd.function.Function):
    """Implements a lif transfer function.

    Note: Uses hand rolled euler integration, might be a better idea to use an existing
    nest/genn implementation.
    """

    @staticmethod
    def forward(ctx, input, weight, lif_parameters, num_timesteps, dt):
        # integrate the lif equations for a given number of timesteps
        voltages, spikes = lif_euler_integrate(input, weight, lif_parameters, num_timesteps, dt)
        # TODO: Need to potentially save LIFParameters as a flat thing here,
        # but we don't use them right now so we should be fine
        ctx.save_for_backward(input, weight, voltages, spikes, torch.tensor(num_timesteps), torch.tensor(dt))
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, voltages, spikes, lif_parameters, num_timesteps, dt = ctx.saved_tensors
        grad_input = grad_weight = grad_voltages = None
        grad_spikes = grad_lif_parameters = grad_num_timesteps = grad_dt = None


        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight, grad_voltages, grad_spikes, grad_lif_parameters, grad_num_timesteps, grad_dt

lif_transfer_function = LIFTransferFunction.apply

class LIF(torch.nn.Module):
    """
    """

    def __init__(self, input_features, output_features, lif_parameters=None, num_timesteps=1000, dt=0.001):
        super(LIF, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1,0.1) # TODO(Christian): Better initialization

        # hyper parameters
        # self.lif_parameters = torch.nn.Parameter(torch.Tensor(output_features, 6), requires_gradient=False)

        if lif_parameters is None:
            self.lif_parameters = constant_lif_parameters(param=default_lif_parameters, size=torch.Size([output_features]))
        else:
            self.lif_parameters = lif_parameters
        self.num_timesteps = num_timesteps
        self.dt = dt


    def forward(self, input):
        return LIFTransferFunction.apply(input, self.weight, self.lif_parameters, self.num_timesteps, self.dt)

    def extra_repr(self):
        return 'in_features={}, out_features={}, lif_parameters={}, num_timesteps={}, dt={}'.format(
            self.in_features, self.out_features, self.lif_parameters.data, self.num_timesteps, self.dt
        )
