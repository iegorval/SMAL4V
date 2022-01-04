import torch
import pickle as pkl
import numpy as np


name2id35 = {
    "RFoot": 14,
    "RFootBack": 24,
    "spine1": 4,
    "Head": 16,
    "LLegBack3": 19,
    "RLegBack1": 21,
    "pelvis0": 1,
    "RLegBack3": 23,
    "LLegBack2": 18,
    "spine0": 3,
    "spine3": 6,
    "spine2": 5,
    "Mouth": 32,
    "Neck": 15,
    "LFootBack": 20,
    "LLegBack1": 17,
    "RLeg3": 13,
    "RLeg2": 12,
    "LLeg1": 7,
    "LLeg3": 9,
    "RLeg1": 11,
    "LLeg2": 8,
    "spine": 2,
    "LFoot": 10,
    "Tail7": 31,
    "Tail6": 30,
    "Tail5": 29,
    "Tail4": 28,
    "Tail3": 27,
    "Tail2": 26,
    "Tail1": 25,
    "RLegBack2": 22,
    "root": 0,
    "LEar": 33,
    "REar": 34,
}
id2name35 = {name2id35[key]: key for key in name2id35.keys()}


class PosePrior:
    def __init__(self, prior_path):
        with open(prior_path, "rb") as f:
            res = pkl.load(f, encoding="latin1")

        # self.precs = res["pic"]
        # self.mean = res["mean_pose"]
        self.precs = torch.from_numpy(res["pic"].r.copy()).float().cuda()
        self.mean = torch.from_numpy(res["mean_pose"].copy()).float().cuda()

        # Mouth closed!
        # self.mean[-2] = -0.4
        # Ignore the first 3 global rotation.
        prefix = 3
        pose_len = 105

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False
        self.use_ind_tch = (
            torch.from_numpy(self.use_ind).float().cuda()
        )  # TODO: add device?

    def __call__(self, x):
        # res = (x[self.use_ind] - self.mean).dot(self.precs)
        # return res

        # Benjamin
        mean_sub = x - self.mean.unsqueeze(0)
        # print("POSE:", self.mean.shape, self.mean.unsqueeze(0).shape)
        res = torch.tensordot(mean_sub, self.precs, dims=([1], [0])) * self.use_ind_tch
        return res ** 2