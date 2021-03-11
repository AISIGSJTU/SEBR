from __future__ import division, print_function
from os import mkdir
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys
import time
import torch.utils.data
from torchvision import transforms, datasets
from torch._six import with_metaclass
from torch._C import _ImperativeEngine as ImperativeEngine

LAMBDA = 0.02
RANDOM_SEED = 42


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


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):
    pass


Variable._execution_engine = ImperativeEngine()


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out


class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return KLD


class BayesLinear_local_reparam(nn.Module):

    def __init__(self, n_in, n_out, prior_sig):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(
            torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

    def forward(self, X, sample=False):
        if not self.training and not sample:  # This is just a placeholder function
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0, 0

        else:
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
            act_W_std = torch.sqrt(torch.mm(X.pow(2), std_w.pow(2)))

            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
            eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1))

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)
            a = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w)
            b = KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)
            kld = a + b
            if LAMBDA < 1e-10:
                lip_loss = 0
            else:
                lip_loss = expectation_spectral_norm_upper_bound_calculation(self.W_mu, self.W_p)
            return output, kld, 0, lip_loss ** 2


class bayes_linear_LR_2L(nn.Module):
    def __init__(self, input_dim, output_dim, nhid, prior_sig):
        super(bayes_linear_LR_2L, self).__init__()

        n_hid = nhid
        self.prior_sig = prior_sig

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bfc1 = BayesLinear_local_reparam(input_dim, n_hid, self.prior_sig)
        self.bfc2 = BayesLinear_local_reparam(n_hid, n_hid, self.prior_sig)
        self.bfc3 = BayesLinear_local_reparam(n_hid, output_dim, self.prior_sig)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, sample=False):
        tlqw = 0
        tlpw = 0
        tlip_loss = 0
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, lqw, lpw, lip_loss = self.bfc1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        tlip_loss += lip_loss
        # -----------------
        x = self.act(x)
        # -----------------
        x, lqw, lpw, lip_loss = self.bfc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        tlip_loss += lip_loss
        # -----------------
        x = self.act(x)
        # -----------------
        y, lqw, lpw, lip_loss = self.bfc3(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        tlip_loss += lip_loss
        return y, tlqw, tlpw, tlip_loss

    def sample_predict(self, x, Nsamples):
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        Hs = []
        for i in range(Nsamples):
            y, tlqw, tlpw, _ = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

            output = nn.functional.softmax(y)
            H = torch.distributions.Categorical(probs=output).entropy()
            Hs.append(H)

        Ha = sum(Hs) / Nsamples
        He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples

        return predictions, tlqw_vec, tlpw_vec, Ha, He


class BBP_Bayes_Net_LR(BaseNet):
    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, Nbatches=0,
                 nhid=1200, prior_sig=0.1):
        super(BBP_Bayes_Net_LR, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.classes = classes
        self.nhid = nhid
        self.prior_sig = prior_sig
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(RANDOM_SEED)
        if self.cuda:
            torch.cuda.manual_seed(RANDOM_SEED)

        self.model = bayes_linear_LR_2L(input_dim=self.channels_in * self.side_in * self.side_in,
                                        output_dim=self.classes,
                                        nhid=self.nhid, prior_sig=self.prior_sig)
        if self.cuda:
            self.model = self.model.cuda()

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()
        lip_loss = 0

        if samples == 1:
            out, tlqw, tlpw, lip_loss = self.model(x)
            mlpdw = F.cross_entropy(out, y, reduction='sum')
            Edkl = (tlqw - tlpw) / self.Nbatches

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tlqw, tlpw, tlip_loss = self.model(x, sample=True)
                mlpdw_i = F.cross_entropy(out, y, reduction='sum')
                Edkl_i = (tlqw - tlpw) / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i
                lip_loss = lip_loss + tlip_loss

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples
            lip_loss = lip_loss / samples

        loss = Edkl + mlpdw + LAMBDA * 0.5 * lip_loss * len(x)
        loss.backward()
        self.optimizer.step()
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return Edkl.data, mlpdw.data, err, 0  # lip_loss.data

    def eval(self, x, y):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _, _ = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _, Ha, He = self.model.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs, Ha, He


if __name__ == "__main__":
    epochs = 50
    prior_sig = 0.1
    lr = 1e-3
    n_samples = 15
    suffix = f"lambda{LAMBDA}_seed{RANDOM_SEED}"
    models_dir = 'models_' + suffix
    results_dir = 'results_' + suffix

    mkdir(models_dir)
    mkdir(results_dir)

    NTrainPointsMNIST = 60000
    batch_size = 100
    nb_epochs = epochs
    log_interval = 1

    cprint('c', '\nData:')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    use_cuda = torch.cuda.is_available()

    trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

    if use_cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)

    cprint('c', '\nNetwork:')

    nsamples = int(n_samples)
    net = BBP_Bayes_Net_LR(lr=lr, channels_in=1, side_in=28, cuda=use_cuda, classes=10, batch_size=batch_size,
                           Nbatches=(NTrainPointsMNIST / batch_size), nhid=1200, prior_sig=prior_sig)

    epoch = 0
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    kl_cost_train = np.zeros(nb_epochs)
    pred_cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    lip_losses = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = np.inf

    nb_its_dev = 1

    train_max_grad = []
    train_mean_grad = []
    test_max_grad = []
    test_mean_grad = []

    tic0 = time.time()

    for i in range(epoch, nb_epochs):
        if i == 0:
            ELBO_samples = nsamples
        else:
            ELBO_samples = nsamples

        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0

        for x, y in trainloader:
            cost_dkl, cost_pred, err, lip_loss = net.fit(x, y, samples=ELBO_samples)
            err_train[i] += err
            kl_cost_train[i] += cost_dkl
            pred_cost_train[i] += cost_pred
            nb_samples += len(x)
            lip_losses[i] += lip_loss

        kl_cost_train[
            i] /= nb_samples
        pred_cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        lip_losses[i] /= nb_samples

        toc = time.time()
        net.epoch = i

        print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, err = %f, lip_loss = %f" % (
            i, nb_epochs, kl_cost_train[i], pred_cost_train[i], err_train[i], lip_losses[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        if i % nb_its_dev == 0:
            net.set_mode_train(False)
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                cost, err, probs = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

                cost_dev[i] += cost
                err_dev[i] += err
                nb_samples += len(x)

            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

            if err_dev[i] < best_err:
                best_err = err_dev[i]
                cprint('b', 'best test error')
                net.save(models_dir + '/theta_best.dat')

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    net.save(models_dir + '/theta_last.dat')

    cprint('c', '\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.min(cost_dev)
    best_cost_train = np.min(pred_cost_train)
    err_dev_min = err_dev[::nb_its_dev].min()

    print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
    print('  err_dev: %f' % (err_dev_min))
    print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
    print('  time_per_it: %fs\n' % (runtime_per_it))
