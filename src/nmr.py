"""
Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""

import pdb
import torch
from src import geom_utils
from neural_renderer.renderer import Renderer


class NeuralRenderer(torch.nn.Module):
    def __init__(
        self,
        img_size=256,
        proj_type="perspective",
        norm_f=1.0,
        norm_z=0.0,
        norm_f0=0.0,
        change_proj_points=True,
    ):
        super(NeuralRenderer, self).__init__()

        # TODO: WHY perspective=False?
        self.renderer = Renderer(
            image_size=img_size,
            light_intensity_ambient=0.8,
            perspective=False,
            camera_mode="look_at",
        )
        self.img_size = img_size
        self.norm_f = norm_f
        self.norm_f0 = norm_f0
        self.norm_z = norm_z
        self.change_proj_points = change_proj_points

        # Set a default camera to be at (0, 0, -1.0)
        self.renderer.eye = [0, 0, -1.0]

        # Silvia
        if proj_type == "perspective":
            self.proj_fn = geom_utils.perspective_proj_withz
        else:
            print("Unknown projection type")
            pdb.set_trace()

        self.offset_z = -1.0

        # Benjamin
        self.textures = (
            torch.ones(7774, 4, 4, 4, 3) * torch.FloatTensor([0, 172, 223]) / 255.0
        ).cuda()  # light blue

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_ambient = 1
        self.renderer.light_intensity_directional = 0

    def directional_light_only(self):
        # Make light only directional.
        self.renderer.light_intensity_ambient = 0.8
        self.renderer.light_intensity_directional = 0.8
        self.renderer.light_direction = [
            0,
            1,
            0,
        ]  # up-to-down, this is the default

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(
            verts,
            cams,
            offset_z=self.offset_z,
            norm_f=self.norm_f,
            norm_z=self.norm_z,
            norm_f0=self.norm_f0,
        )

        # Silvia
        proj_points = proj[:, :, :2]
        proj_points = (proj_points[:, :, :2] + 1) * 0.5 * self.img_size

        if self.change_proj_points:  # Benjamin
            proj_points = torch.stack(
                [
                    proj_points[:, :, 0],
                    self.img_size - proj_points[:, :, 1],
                ],
                dim=-1,
            )
        return proj_points

    def forward(self, vertices, faces, cams, gpu_id, textures=None):  # TODO: gpu_id
        verts = self.proj_fn(
            vertices,
            cams,
            offset_z=self.offset_z,
            norm_f=self.norm_f,
            norm_z=self.norm_z,
            norm_f0=self.norm_f0,
        )

        if textures is not None:
            return self.renderer(verts, faces, textures, mode="rgb")
        else:
            # Benjamin
            textures = self.textures.cuda(device=gpu_id)
            textures = textures.unsqueeze(0).expand(verts.shape[0], -1, -1, -1, -1, -1)
            img = self.renderer.render_rgb(verts, faces, textures)
            sil = self.renderer.render_silhouettes(verts, faces)
            return img, sil
            # return self.renderer(verts, faces, mode="silhouettes")
