import torch

class PolicyValueNetwork(torch.nn.Module):

    def __init__(self, action_space, obs_space):
        super(PolicyValueNetwork, self).__init__()

        self.obs_space = obs_space

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.obs_space, 1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, action_space),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=-1)
        )

        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(self.obs_space, 1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.ReLU()
        )

    def action(self, x):
        x = x.to(next(self.policy_net.parameters()).device)
        log_probs = self.policy_net(x)
        distrib = torch.distributions.Categorical(log_probs)
        act = distrib.sample()
        return act.item(), distrib.log_prob(act)

    def value(self, x):
        x = x.to(next(self.policy_net.parameters()).device)
        value = self.value_net(x)
        return value