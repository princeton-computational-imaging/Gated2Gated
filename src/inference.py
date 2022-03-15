from __future__ import absolute_import, division, print_function
import visualize2D
import networks

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import PIL.Image as pil
import matplotlib.cm as cm
import numpy as np
from layers import disp_to_depth



import torch
from torchvision import transforms
to_tensor = transforms.ToTensor()

cmap_dict = {
    'jet': cm.jet,
    'jet_r': cm.jet_r,
    'plasma': cm.plasma,
    'plasma_r': cm.plasma_r,
    'magma': cm.magma,
    'magma_r': cm.magma_r,
    'inferno': cm.inferno,
    'inferno_r': cm.inferno_r
}

def read_gated_image(base_dir, img_id, num_bits=10, data_type='real',
                     scale_images=False, scaled_img_width=None, crop_size_h=104, crop_size_w=128, scaled_img_height=None):

    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir, 'gated%d_10bit' % gate_id)
        path = os.path.join(gate_dir, img_id + '.png')
        assert os.path.exists(path), "No such file : %s" % path
        img = cv2.imread(os.path.join(
            gate_dir, img_id + '.png'), cv2.IMREAD_UNCHANGED)
        if data_type == 'real':
            img = img[crop_size_h:(img.shape[0] - crop_size_h),
                      crop_size_w:(img.shape[1] - crop_size_w)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer

        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width,
                         scaled_img_height), interpolation=cv2.INTER_AREA)
    return img


def load_weights(model, pretrained_weights_path):
    model_dict = model.state_dict()
    assert os.path.isfile(pretrained_weights_path), "{} not found in the location".format(
        os.path.basename(pretrained_weights_path))
    pretrained_dict = torch.load(pretrained_weights_path)
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_depth_viz(depthmap, save_path, min_depth, max_depth, colormap):
    # Generate colorized depth map
    depth_map_color = visualize2D.colorize_depth(
        depthmap, min_distance=min_depth, max_distance=max_depth, cmap=colormap)
    depth_map_color = pil.fromarray(depth_map_color.astype(np.uint8))
    depth_map_color.save(save_path)


def inference(options):

    models = {}

    models["depth"] = networks.PackNetSlim01(
        dropout=0.5, version="{}{}".format(1, 'A'))
    models["depth"].to('cuda')

    models["encoder"] = networks.Encoder(num_convs=4)
    models["encoder"].to('cuda')

    models["albedo"] = networks.Decoder(
        name="albedo", scales=range(1), out_channels=1)
    models["albedo"].to('cuda')

    models["ambient"] = networks.Decoder(
        name="ambient", scales=range(1), out_channels=1)
    models["ambient"].to('cuda')

    # Load model weights
    models["depth"] = load_weights(
        models["depth"], os.path.join(options.weights_dir, "depth.pth"))
    models["encoder"] = load_weights(
        models["encoder"], os.path.join(options.weights_dir, "encoder.pth"))
    models["albedo"] = load_weights(
        models["albedo"], os.path.join(options.weights_dir, "albedo.pth"))
    models["ambient"] = load_weights(
        models["ambient"], os.path.join(options.weights_dir, "ambient.pth"))

    # Eval Mode
    for model in models.values():
        model.eval()

    results_dirs = ["depth", "ambient", "albedo"]
    for _dir in results_dirs:
        os.makedirs(os.path.join(options.results_dir, _dir), exist_ok=True)

    imgs_names = [sample for sample in os.listdir(os.path.join(options.data_dir, "gated0_10bit")) if '.png' in sample]
    img_ids = list(map(lambda x: x.split('.')[0], imgs_names))

    with torch.no_grad():
        for img_id in img_ids:
            gated_img = to_tensor(read_gated_image(
                options.data_dir, img_id)).unsqueeze(0).to('cuda')

            # Getting depth
            disp = models['depth'](gated_img)[('disp', 0)]
            _, pred_depth = disp_to_depth(
                disp, options.min_depth, options.max_depth)
            pred_depth = pred_depth[0, 0].cpu(
            ).numpy() * options.depth_normalizer
            pred_depth = np.clip(pred_depth, 0.0, options.clip_depth)
            np.savez(os.path.join(options.results_dir, "depth",
                     "{}.npz".format(img_id)), pred_depth)
            save_depth_viz(pred_depth,os.path.join(options.results_dir, "depth",
                     "{}.png".format(img_id)), 0.0, options.clip_depth,
                           cmap_dict["inferno_r"])

            feats = models['encoder'](gated_img)

            # Getting ambient
            _ambient = models['ambient'](feats)[('ambient', 0)]
            ambient = _ambient[0, 0].cpu().numpy()
            ambient = np.clip(ambient, 0.0, 1.0) * 255.
            ambient = pil.fromarray(ambient.astype(np.uint8))
            ambient.save(os.path.join(options.results_dir, "ambient",
                     "{}.png".format(img_id)))

            # Getting albedo
            _albedo = models['albedo'](feats)[('albedo', 0)]
            albedo = _albedo[0, 0].cpu().numpy()
            albedo = np.clip(albedo, 0.0, 1.0) * 255.
            albedo = pil.fromarray(albedo.astype(np.uint8))
            albedo.save(os.path.join(options.results_dir, "albedo",
                                      "{}.png".format(img_id)))


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--data_dir", required=True,
                         help="Path to the dataset directory")
    options.add_argument("--min_depth", default=0.1,
                         type=float,
                         help="Minimum depth value to evaluate")
    options.add_argument("--max_depth", default=100.0,
                         type=float,
                         help="Max depth value to evaluate")
    options.add_argument("--clip_depth", default=80.0,
                         type=float,
                         help="clip depth to this value")
    options.add_argument("--height", default=512,
                         type=int,
                         help="height of crop for gated image")
    options.add_argument("--width", default=1024,
                         type=int,
                         help="width of crop for gated image")
    options.add_argument("--depth_normalizer", default=70.0,
                         type=float,
                         help="depth normalizer to multiply predicted depth with")
    options.add_argument("--weights_dir", required=True,
                         help="Path where weights are stored")
    options.add_argument("--results_dir", required=True,
                         help="Path where results are stored")
    options.add_argument("--cmap", default='inferno_r',
                         choices=['jet', 'jet_r', 'plasma', 'plasma_r',
                                  'magma', 'magma_r', 'inferno', 'inferno_r'],
                         help="Which colormap to use for generating results")

    options = options.parse_args()
    inference(options)
