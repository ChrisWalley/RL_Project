import torch

class PolicyValueNetwork(torch.nn.Module):

    def __init__(self, action_space, alpha, obs_space):
        super(PolicyValueNetwork, self).__init__()

        self.obs_space = obs_space

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.obs_space, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, action_space),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=-1)
        )

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = alpha)

        #self.value_net = torch.nn.Sequential(
        #    torch.nn.Linear(self.obs_space, self.obs_space*2),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(self.obs_space*2, self.obs_space),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(self.obs_space, 1)
        #)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(self.obs_space, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(64, self.obs_space),
            torch.nn.ReLU(),
            torch.nn.Linear(self.obs_space, 1),
        )

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr = alpha)

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