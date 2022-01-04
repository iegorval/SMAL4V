"""
Generic Training Utils.

Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""

import torch
import os
import time
import pdb
import time
import imageio
import pickle as pkl
import numpy as np
import os.path as osp
from src import geom_utils
from absl import flags
from utils.visualizer import Visualizer
from src.smal_mesh_eval import smal_mesh_eval


def set_bn_eval(m):
    classname = m.__class__.__name__
    if (classname.find("BatchNorm1d") != -1) or (classname.find("BatchNorm2d") != -1):
        m.eval()


# -------- tranining class ---------#
# ----------------------------------#
class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts["gpu_id"]
        self.Tensor = (
            torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        )
        self.invalid_batch = False  # the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts["checkpoint_dir"], opts["name"])
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, "opts.log")
        with open(log_file, "w") as f:
            for k in opts.keys():
                f.write("{}: {}\n".format(k, opts[k]))

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = "{}_net_{}.pth".format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = "{}_net_{}.pth".format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def define_model(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def init_dataset(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def define_criterion(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def set_input(self, batch):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def forward(self):
        """Should compute self.total_loss. To be implemented by the child class."""
        raise NotImplementedError

    def save(self, epoch_prefix):
        """Saves the model."""
        self.save_network(self.model, "pred", epoch_prefix, gpu_id=self.gpu_id)
        return

    def get_current_visuals(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def get_current_scalars(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def get_current_points(self):
        """Should be implemented by the child class."""
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        if opts["use_sgd"]:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opts["learning_rate"],
                momentum=opts["beta1"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opts["learning_rate"],
                betas=(opts["beta1"], 0.999),
            )

    def save_current(self, opts, initial_loss=0, final_loss=0, code="_"):
        res_dict = {
            "final_loss": final_loss.data.detach().cpu().numpy(),
            "delta_v": self.delta_v.data.detach().cpu().numpy(),
            "kp_pred": self.kp_pred.data.detach().cpu().numpy(),
            "scale": self.scale_pred.data.detach().cpu().numpy(),
            "trans": self.trans_pred.data.detach().cpu().numpy(),
            "pose": self.pose_pred.data.detach().cpu().numpy(),
            "initial_loss": initial_loss.data.detach().cpu().numpy(),
        }
        img_ext = "." + opts["image_file_string"].split(".")[-1]  # TODO: REWRITE
        imageio.imwrite(
            opts["image_file_string"].replace(img_ext, code + "mask.png"),
            (255 * self.mask_pred.detach().cpu()[0, :, :]),
        )
        uv_flows = self.model.texture_predictor.uvimage_pred
        uv_flows = uv_flows.permute(0, 2, 3, 1)
        uv_images = torch.nn.functional.grid_sample(
            self.imgs, uv_flows, align_corners=True
        )

        texname = opts["image_file_string"].replace(img_ext, code + "tex.png")
        imageio.imwrite(
            texname,
            (255 * np.transpose(uv_images.detach().cpu()[0, :, :, :], (1, 2, 0))),
        )
        fname = opts["image_file_string"].replace(img_ext, code + "res.pkl")
        pkl.dump(res_dict, open(fname, "wb"))

    def train(self):
        # time_stamp = str(time.time())[:10]
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer

        total_steps = 0
        dataset_size = len(self.dataloader)
        print("dataset_size " + str(dataset_size))

        v_log_file = os.path.join(self.save_dir, "validation.log")
        curr_epoch_err = 1000

        if opts["is_optimization"]:
            self.model.eval()
            self.model.texture_predictor.eval()
            for param in self.model.texture_predictor.parameters():
                param.requires_grad = False

            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            self.model.code_predictor.shape_predictor.pred_layer.eval()
            for (
                param
            ) in self.model.code_predictor.shape_predictor.pred_layer.parameters():
                param.requires_grad = False

            self.model.code_predictor.shape_predictor.fc.eval()
            for param in self.model.code_predictor.shape_predictor.fc.parameters():
                param.requires_grad = False

            self.model.code_predictor.scale_predictor.pred_layer.eval()
            for (
                param
            ) in self.model.code_predictor.scale_predictor.pred_layer.parameters():
                param.requires_grad = False

            self.model.code_predictor.trans_predictor.pred_layer_xy.eval()
            for (
                param
            ) in self.model.code_predictor.trans_predictor.pred_layer_xy.parameters():
                param.requires_grad = False

            self.model.code_predictor.trans_predictor.pred_layer_z.eval()
            for (
                param
            ) in self.model.code_predictor.trans_predictor.pred_layer_z.parameters():
                param.requires_grad = False

            self.model.code_predictor.pose_predictor.pred_layer.eval()
            for (
                param
            ) in self.model.code_predictor.pose_predictor.pred_layer.parameters():
                param.requires_grad = False

            self.model.apply(set_bn_eval)

            code = osp.splitext(osp.basename(opts["image_file_string"]))[0]
            visualizer.print_message(code)

        self.background_model_top = None
        set_optimization_input = True
        for epoch in range(opts["num_pretrain_epochs"], opts["num_epochs"]):
            epoch_iter = 0
            for batch in self.dataloader:
                iter_start_time = time.time()
                if not opts["is_optimization"]:
                    self.set_input(batch)
                else:
                    if set_optimization_input:
                        self.set_input(batch)

                if not self.invalid_batch:
                    self.optimizer.zero_grad()
                    self.forward()

                    if opts["is_optimization"]:
                        if set_optimization_input:
                            initial_loss = self.tex_loss
                            print("Initial loss")
                            print(initial_loss)
                            current_loss = initial_loss
                            opt_loss = current_loss
                            # Now the input should be the image prediction
                            self.set_optimization_input()
                            set_optimization_input = False
                            self.save_current(
                                opts, initial_loss, current_loss, code="_init_"
                            )
                        else:
                            current_loss = self.tex_loss
                            if current_loss < opt_loss:
                                opt_loss = current_loss
                                self.save_current(
                                    opts, initial_loss, current_loss, code="_best_"
                                )
                                visualizer.print_message(
                                    "save current best " + str(current_loss)
                                )

                    # self.background_model_top is not used but exloited as a flag
                    if opts["is_optimization"] and self.background_model_top is None:
                        # Create background model with current prediction
                        M = np.abs(self.mask_pred.cpu().detach().numpy()[0, :, :] - 1)
                        I = np.transpose(
                            self.imgs.cpu().detach().numpy()[0, :, :, :], (1, 2, 0)
                        )
                        N = 128

                        # Top half of the image
                        self.background_model_top = np.zeros((3))
                        n = np.sum(M[:N, :])
                        for c in range(3):
                            J = I[:, :, c] * M
                            self.background_model_top[c] = np.sum(J[:N, :]) / n

                        self.background_model_bottom = np.zeros((3))
                        n = np.sum(M[N:, :])
                        for c in range(3):
                            J = I[:, :, c] * M
                            self.background_model_bottom[c] = np.sum(J[N:, :]) / n
                        if opts["use_sgd"]:
                            self.optimizer = torch.optim.SGD(
                                [self.model.op_features],
                                lr=opts["learning_rate"],
                                momentum=opts["beta1"],
                            )
                        else:
                            if opts["is_var_opt"]:
                                self.optimizer = torch.optim.Adam(
                                    self.model.op_features,
                                    lr=opts["learning_rate"],
                                    betas=(opts["beta1"], 0.999),
                                )
                            else:
                                self.optimizer = torch.optim.Adam(
                                    [self.model.op_features],
                                    lr=opts["learning_rate"],
                                    betas=(opts["beta1"], 0.999),
                                )

                    self.smoothed_total_loss = (
                        self.smoothed_total_loss * 0.99 + 0.01 * self.total_loss.data
                    )
                    self.total_loss.backward()
                    self.optimizer.step()

                total_steps += 1
                epoch_iter += 1

                if opts["display_visuals"] and (
                    total_steps % opts["display_freq"] == 0
                ):
                    iter_end_time = time.time()
                    print(
                        "time/itr %.2g"
                        % ((iter_end_time - iter_start_time) / opts["display_freq"])
                    )
                    visualizer.display_current_results(
                        self.get_current_visuals(), epoch
                    )
                    visualizer.plot_current_points(self.get_current_points())

                if opts["print_scalars"] and (total_steps % opts["print_freq"] == 0):
                    scalars = self.get_current_scalars()
                    visualizer.print_current_scalars(epoch, epoch_iter, scalars)
                    if opts["plot_scalars"]:
                        visualizer.plot_current_scalars(
                            epoch, float(epoch_iter) / dataset_size, opts, scalars
                        )

                if total_steps % opts["save_latest_freq"] == 0:
                    print(
                        "saving the model at the end of epoch {:d}, iters {:d}".format(
                            epoch, total_steps
                        )
                    )
                    self.save("latest")

                if total_steps == opts["num_iter"]:
                    return

            if opts["do_validation"]:
                self.save("100000")  # TODO: What is that?
                epoch_err = smal_mesh_eval(num_train_epoch=100000, opts=self.opts)
                if epoch_err <= curr_epoch_err:
                    print("update best model")
                    curr_epoch_err = epoch_err
                    self.save("best")
                    with open(v_log_file, "a") as f:
                        f.write("{}: {}\n".format(epoch, epoch_err))

            if (epoch + 1) % opts["save_epoch_freq"] == 0:
                print(
                    "saving the model at the end of epoch {:d}, iters {:d}".format(
                        epoch, total_steps
                    )
                )
                if opts["is_optimization"]:
                    self.save_current(opts, initial_loss, current_loss, code=None)
                else:
                    self.save(epoch + 1)
                    self.save("latest")
            if opts["is_optimization"]:
                if opt_loss < initial_loss:
                    visualizer.print_message("updated")
