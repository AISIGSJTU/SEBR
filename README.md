# SEBR
This is the official repository for paper *Robust Bayesian Neural Networks by Spectral Expectation Bound Regularization*, accepted by CVPR 2021 as a poster paper.

# Core Algorithm

The algorithm for the calculation of SEBR loss is:

    def expectation_spectral_norm_upper_bound_calculation(W_mu, W_p=None, SIMU_TIMES=10, ITERATION_TIMES=10):
        u = torch.rand(W_mu.shape[0]).cuda()
        v = torch.rand(W_mu.shape[1]).cuda()
        for _ in range(ITERATION_TIMES):
            v = torch.nn.functional.normalize(torch.mv(W_mu.t(), u), dim=0, eps=1e-12)
            u = torch.nn.functional.normalize(torch.mv(W_mu, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.mv(W_mu, v))
        if W_p is None:
            return sigma

        std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

        res = torch.max(torch.norm(std_w, dim=1)) + torch.max(torch.norm(std_w, dim=0))

        tmp = 0
        for _ in range(SIMU_TIMES):
            eps_W = W_mu.data.new(W_mu.size()).normal_()
            tmp += torch.max(1 * eps_W * std_w)
        tmp /= SIMU_TIMES
        return res + tmp + sigma
