from os.path import join
import pickle as pkl
import numpy as np
import torch

SMAL_N_BETAS = 26


class MultiShapePrior:
    def __init__(self, family_name, data_path):
        with open(data_path, "rb") as f:
            data = pkl.load(f, encoding="latin1")
        if family_name == "tiger" or family_name == "big_cats":
            family_id = 0
        elif family_name == "dog":
            family_id = 1
        elif family_name == "horse":
            family_id = 2
        elif family_name == "cow":
            family_id = 3
        elif family_name == "hippo":
            family_id = 4
        else:
            print("Dont know animal %s!" % family_name)

        mean = data["cluster_means"][family_id]
        if "cluster_precs" in data:
            prec = data["cluster_precs"][family_id]
        else:
            cov = data["cluster_cov"][family_id]
            invcov = np.linalg.pinv(cov + 1e-5 * np.eye(cov.shape[0]))
            prec = np.linalg.cholesky(invcov)

        self.mean = torch.from_numpy(mean).float().cuda()
        self.precs = torch.from_numpy(prec).float().cuda()

    def __call__(self, x):
        # Mahalanobis.
        # return (x - self.mu[: len(x)]).dot(self.prec[: len(x), : len(x)])
        mean = self.mean[: x.shape[1]]
        # print(mean.shape, mean.unsqueeze(0).shape, x.shape)
        precs = self.precs[: x.shape[1], : x.shape[1]]
        mean_sub = x - mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, precs, dims=([1], [0]))
        return res ** 2
