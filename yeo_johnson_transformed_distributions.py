import torch
from torch.distributions import Normal, Transform, constraints
from torch.distributions.transforms import AffineTransform, PowerTransform, ExpTransform, ComposeTransform
from torch.distributions.transformed_distribution import TransformedDistribution


class YeoJohnsonTransform(Transform):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, lbda):
        super().__init__()
        self.lbda = lbda

        # Positive domain transforms
        if self.lbda == 0:
            self.positive_transform = ComposeTransform([
                AffineTransform(loc=1., scale=1.),
                ExpTransform().inv,
            ])
        else:
            self.positive_transform = ComposeTransform([
                AffineTransform(loc=1., scale=1.),
                PowerTransform(self.lbda),
                AffineTransform(loc=-1., scale=1.),
                AffineTransform(loc=0., scale=1. / self.lbda),
            ])

        # Negative domain transforms
        if self.lbda == 2:
            self.negative_transform = ComposeTransform([
                AffineTransform(loc=1., scale=-1.),
                ExpTransform().inv,
                AffineTransform(loc=1., scale=-1.),
            ])
        else:
            self.negative_transform = ComposeTransform([
                AffineTransform(loc=1., scale=-1.),
                PowerTransform(2. - self.lbda),
                AffineTransform(loc=-1., scale=1.),
                AffineTransform(loc=0., scale=-1. / (2. - self.lbda)),
            ])

    def _call(self, x):
        positive_mask = x >= 0.
        negative_mask = ~positive_mask

        y = torch.zeros_like(x)
        y[positive_mask] = self.positive_transform(x[positive_mask])
        y[negative_mask] = self.negative_transform(x[negative_mask])

        return y

    @property
    def exponent(self):
        return self.lbda


class YeoJohnsonNormal(TransformedDistribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "tloc": constraints.real,
        "tscale": constraints.real,
        "lbda": constraints.real
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, scale, lbda, tloc, tscale, validate_args=None):
        base_distribution = Normal(loc, scale)

        self.yj_transform = YeoJohnsonTransform(lbda=lbda)

        self.scalar_transform = AffineTransform(loc=tloc, scale=tscale)

        transforms = [
            self.yj_transform,
            self.scalar_transform,
        ]

        super().__init__(base_distribution, transforms, validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def tloc(self):
        return self.transforms[1].loc

    @property
    def tscale(self):
        return self.transforms[1].scale

    @property
    def lbda(self):
        return self.transforms[0].exponent


if __name__ == "__main__":
    from sklearn.preprocessing import PowerTransformer
    import matplotlib.pyplot as plt

    n_samples = 1_000_000

    dist = Normal(-5., 0.5)
    dist_samples = dist.sample((n_samples,)).view(-1, 1)

    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True, )
    power_transformer.fit(dist_samples.cpu().numpy())
    pt_normal_samples = power_transformer.transform(dist_samples.cpu().numpy())
    rec_pt_normal_samples = power_transformer.inverse_transform(pt_normal_samples)

    tloc = torch.tensor(power_transformer._scaler.mean_)
    tscale = torch.tensor(power_transformer._scaler.scale_)
    lbda = torch.tensor(power_transformer.lambdas_)

    yjnormal = YeoJohnsonNormal(
        loc=torch.tensor([0.]),
        scale=torch.tensor([1.]),
        lbda=lbda,
        tloc=tloc,
        tscale=tscale
    )

    samples = yjnormal.sample((int(n_samples),))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=True)

    ax0.hist(pt_normal_samples, bins=50, density=True,
            label="sklearn YJT normalized", alpha=0.3)

    ax0.hist(Normal(0., 1).sample((n_samples,)).view(-1, 1).cpu().numpy(), bins=50, density=True,
             label="Standard normal", alpha=0.3)

    ax0.legend()
    ax0.set_ylabel("Density")
    ax0.set_xlabel("X")
    
    x = torch.linspace(-10, 10, 1000)

    # ax1.hist(dist_samples.cpu().numpy(), bins=50, density=True,
    #         label="Data", alpha=0.3)

    ax1.plot(x.cpu().numpy(),
            dist.log_prob(x).exp().cpu().numpy(),
            label="Data PDF", ls="dashed")

    ax1.hist(samples.cpu().numpy(), bins=50, density=True,
            label="Yeo-Johnson Transformed Normal", alpha=0.3)

    ax1.hist(rec_pt_normal_samples, bins=50, density=True,
            label="sklearn YJT reconstructed", alpha=0.3)

    ax1.set_xlabel("X")
    ax1.legend()

    plt.show()





