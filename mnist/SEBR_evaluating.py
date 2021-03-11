from __future__ import division, print_function
import torch.utils.data
from SEBR_training import *


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


use_cuda = torch.cuda.is_available()
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
batch_size = 100
NTrainPointsMNIST = 60000
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=3)
net = BBP_Bayes_Net_LR(channels_in=1, side_in=28, cuda=use_cuda, classes=10, batch_size=batch_size,
                       Nbatches=(NTrainPointsMNIST // batch_size), nhid=1200)
MNIST_normalize = Normalize(mean=(0.1307,), std=(0.3081,))


def fgsm(model, X, y, norm, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    X2 = norm(X + delta).cuda()
    outputs = model(X2)
    if type(outputs) == type(()):
        outputs = outputs[0]
    loss = nn.CrossEntropyLoss()(outputs, y.cuda())
    loss.backward()
    return epsilon * delta.grad.sign()


def pgd(model, X, y, norm, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    delta.data.uniform_(-epsilon, epsilon)
    for t in range(num_iter):
        X2 = norm(X + delta).cuda()
        outputs = model(X2)
        if type(outputs) == type(()):
            outputs = outputs[0]
        loss = nn.CrossEntropyLoss()(outputs, y.cuda())
        loss.backward()
        delta.data = (delta + alpha * delta.grad.data.sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def test_with_adv_noise(attack, noise_ratios):
    assert attack in ['pgd', 'fgsm']
    cost_dev = 0.0
    err_dev = 0.0
    errs = []
    for noise_ratio in noise_ratios:
        nb_samples = 0.0
        for j, (x, y) in enumerate(valloader):
            if attack == 'pgd':
                x_noise = x + pgd(net.model, x, y, MNIST_normalize, noise_ratio, 1e5, 15)
            elif attack == 'fgsm':
                x_noise = x + fgsm(net.model, x, y, MNIST_normalize, noise_ratio)
            else:
                print("Undefined attack method!")
                exit()
            cost, err, probs, _, _ = net.sample_eval(MNIST_normalize(x_noise), y, Nsamples=5)

            err_dev += err
            cost_dev += cost
            nb_samples += len(x)

        cost_dev /= nb_samples
        err_dev /= nb_samples

        errs.append(float(err_dev))

        cprint('g', ' Jdev = %f, err = %f\n' % (cost_dev, err_dev))

    return errs


if __name__ == '__main__':
    attack = 'fgsm'
    LAMBDA = 0.02
    RANDOM_SEED = 42
    noise_ratios = [0, 0.04, 0.16, 0.3]
    suffix = f"lambda{LAMBDA}_seed{RANDOM_SEED}"
    models_dir = 'models_' + suffix
    path = models_dir + "/theta_last.dat"
    net.load(path)
    net.set_mode_train(False)
    err = test_with_adv_noise(attack=attack, noise_ratios=noise_ratios)
    print("The error rates are: ")
    print(err)
