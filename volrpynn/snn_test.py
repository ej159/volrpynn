import snn
import torch

def test_default_parameters():
    # this test is here because we assume certain
    # constant values below

    param=snn.default_lif_parameters

    assert(param.v_th == 75)
    assert(param.v_rest == 50)
    assert(param.v_reset == 30)
    assert(param.t_ref == 0.001)
    assert(param.tau == 2)

def test_heavy_side():
    x = torch.tensor([1.0])
    h = torch.tensor([0.1])
    one = torch.ones(1)
    zero = torch.zeros(1)

    assert(snn.heavy_side(x, h) == one)
    assert(snn.heavy_side(-x, h) == zero)

    # TODO: Check expected behaviour of gradient

def test_lif_step():
    parameters = snn.constant_lif_parameters(param=snn.default_lif_parameters, size=torch.Size([1]))

    v, spike, _ = snn.lif_step(
        v = parameters.v_rest.clone().detach(),
        x = torch.tensor([1.0]),
        refrac_count = torch.tensor([0.0]),
        parameters = parameters,
        dt = 0.001
    )

    assert(v == parameters.v_rest + 1.0)


    v, spike, _ = snn.lif_step(
        v = parameters.v_rest.clone().detach(),
        x = torch.tensor([0.0]),
        refrac_count = torch.tensor([0.0]),
        parameters = parameters,
        dt = 0.001
    )

    assert(v == parameters.v_rest)

    v, spike, _ = snn.lif_step(
        v = parameters.v_rest.clone().detach() + 1.0,
        x = torch.tensor([0.0]),
        refrac_count = torch.tensor([0.0]),
        parameters = parameters,
        dt = 0.001
    )

    assert(v == 50.9980)

    refrac_count = torch.tensor([0.0])
    v, spike, refrac_count = snn.lif_step(
        v = parameters.v_th.clone().detach() - 0.8,
        x = torch.tensor([1.0]),
        refrac_count = refrac_count,
        parameters = parameters,
        dt = 0.001
    )

    assert(v == parameters.v_reset)
    assert(spike == torch.tensor([1.0]))
    assert(refrac_count == parameters.t_ref)

def test_adaptive_lif_step():
    parameters = snn.constant_adaptive_lif_parameters(param=snn.default_adaptive_lif_parameters, size=torch.Size([1]))
    v, spike, _, beta = snn.adaptive_lif_step(
        v = parameters.v_rest.clone().detach(),
        x = torch.tensor([1.0]),
        beta = torch.tensor([0.0]),
        refrac_count = torch.tensor([0.0]),
        parameters = parameters,
        dt = 0.001
    )

    assert(v == parameters.v_rest + 1.0)

def test_lif_euler_integrate():
    torch.manual_seed(42)
    n_timesteps = 1000
    dist = torch.distributions.bernoulli.Bernoulli(0.1 * torch.ones(n_timesteps,1))
    x = dist.sample()
    weight = torch.tensor([[3.0]])
    parameters = snn.constant_lif_parameters(param=snn.default_lif_parameters, size=torch.Size([1]))

    v,s = snn.lif_euler_integrate(x, weight, lif_parameters=parameters, num_timesteps=n_timesteps, dt=0.001)

    # import matplotlib.pyplot as plt
    # plt.plot(v.numpy())
    # plt.show()

    # assert number of seen spikes
    assert(torch.sum(s) == 7)

def test_lif_euler_integrate_backward():
    torch.manual_seed(42)
    n_timesteps = 1000
    dist = torch.distributions.bernoulli.Bernoulli(0.1 * torch.ones(n_timesteps,1))
    x = dist.sample()

    weight = torch.tensor([[3.0]], requires_grad=True)
    parameters = snn.constant_lif_parameters(param=snn.default_lif_parameters, size=torch.Size([1]))

    optimizer = torch.optim.Adam([weight])

    num_target_spikes = 8

    # ensure that we are learning something
    v,s = snn.lif_euler_integrate(x, weight, lif_parameters=parameters, num_timesteps=n_timesteps, dt=0.001)
    assert(torch.sum(s) != num_target_spikes) # sanity

    vs = []

    for i in range(400):
        optimizer.zero_grad()
        v,s = snn.lif_euler_integrate(x, weight, lif_parameters=parameters, num_timesteps=n_timesteps, dt=0.001)
        loss = (torch.sum(s) - num_target_spikes)**2
        loss.backward()
        # if i % 100 == 0:
        #     print(i, weight.grad, weight.data, loss.data)
        optimizer.step()
        vs.append(v)


    assert(torch.sum(s) == num_target_spikes)


def test_lif_transfer_function():
    # this is basically the same as test_lif_euler_integration, except we
    # also use the pytorch machinery
    torch.manual_seed(42)
    n_timesteps = 1000
    dist = torch.distributions.bernoulli.Bernoulli(0.1 * torch.ones(n_timesteps,1))
    x = dist.sample()
    weight = torch.tensor([[3.0]])
    parameters = snn.constant_lif_parameters(param=snn.default_lif_parameters, size=torch.Size([1]))

    s = snn.lif_transfer_function(x, weight, parameters, n_timesteps, 0.001)
    # sanity check
    v_,s_ = snn.lif_euler_integrate(x, weight, lif_parameters=parameters, num_timesteps=n_timesteps, dt=0.001)

    assert(torch.sum(s) == torch.sum(s_))

def test_lif_module():
    torch.manual_seed(42)
    n_timesteps = 1000
    dist = torch.distributions.bernoulli.Bernoulli(0.1 * torch.ones(n_timesteps,1))
    x = dist.sample()
    m = snn.LIF(1,1)
    m.weight.data = torch.tensor([[3.0]])

    s = m(x)

    assert(torch.sum(s).item() == 7)


if __name__ == "__main__":
    test_default_parameters()
    test_heavy_side()
    test_lif_step()
    test_adaptive_lif_step()
    test_lif_euler_integrate()
    test_lif_transfer_function()
    test_lif_module()
    # the following tests take a long time
    test_lif_euler_integrate_backward()
