import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))     #inp=[41, 10, 10, 7]-->[41, 10, 10, 3]-->[4100, 3]
    mask = inp[..., 6].reshape(-1)          #inp=[41, 10, 10, 7]-->[41, 10, 10]-->[4100], mask has 2998 1's and 1112 0's
    point_indices_inside_faces = mask == 1  #[4100]=tensor([ True, False, False,...]) has 2998 True and 1112 False
    pts = pts[point_indices_inside_faces, :]#Activate the row values of 1st, 2nd, and 3rd column according to mask "point_indices_inside_faces" 
    return bounding_box_pointcloud(pts)     #pts.shape=[2988, 3]


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]  #torch.tensor(box).shape=[2, 3]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]  #diag=XYZ_range tensor=max-min  #shape=[3]
    scale = 2.0 / max(diag[0], diag[1], diag[2])  #scale=2.0/max(XYZ_range)
    center = 0.5 * (bbox[0] + bbox[1]) #center=XYZ_center=0.5(min+max)   #shape=[3]
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def get_random_rotation():
    """Get a random rotation in 90 degree increments along the canonical axes"""
    axes = [np.array([1, 0, 0]), 
            np.array([0, 1, 0]), 
            np.array([0, 0, 1])]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """Rotate the node features in the graph by a given rotation"""
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(orig_size)  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(orig_size)  # Normals/tangents
    return inp


INVALID_FONTS = [
    "Bokor",
    "Lao Muang Khong",
    "Lao Sans Pro",
    "MS Outlook",
    "Catamaran Black",
    "Dubai",
    "HoloLens MDL2 Assets",
    "Lao Muang Don",
    "Oxanium Medium",
    "Rounded Mplus 1c",
    "Moul Pali",
    "Noto Sans Tamil",
    "Webdings",
    "Armata",
    "Koulen",
    "Yinmar",
    "Ponnala",
    "Noto Sans Tamil",
    "Chenla",
    "Lohit Devanagari",
    "Metal",
    "MS Office Symbol",
    "Cormorant Garamond Medium",
    "Chiller",
    "Give You Glory",
    "Hind Vadodara Light",
    "Libre Barcode 39 Extended",
    "Myanmar Sans Pro",
    "Scheherazade",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Signika SemiBold" "Taprom",
    "Times New Roman TUR",
    "Playfair Display SC Black",
    "Poppins Thin",
    "Raleway Dots",
    "Raleway Thin",
    "Segoe MDL2 Assets",
    "Segoe MDL2 Assets",
    "Spectral SC ExtraLight",
    "Txt",
    "Uchen",
    "Yinmar",
    "Almarai ExtraBold",
    "Fasthand",
    "Exo",
    "Freckle Face",
    "Montserrat Light",
    "Inter",
    "MS Reference Specialty",
    "MS Outlook",
    "Preah Vihear",
    "Sitara",
    "Barkerville Old Face",
    "Bodoni MT" "Bokor",
    "Fasthand",
    "HoloLens MDL2 Assests",
    "Libre Barcode 39",
    "Lohit Tamil",
    "Marlett",
    "MS outlook",
    "MS office Symbol Semilight",
    "MS office symbol regular",
    "Ms office symbol extralight",
    "Ms Reference speciality",
    "Segoe MDL2 Assets",
    "Siemreap",
    "Sitara",
    "Symbol",
    "Wingdings",
    "Metal",
    "Ponnala",
    "Webdings",
    "Souliyo Unicode",
    "Aguafina Script",
    "Yantramanav Black",
    # "Yaldevi",
    # Taprom,
    # "Zhi Mang Xing",
    # "Taviraj",
    # "SeoulNamsan EB",
]


def valid_font(filename):
    for name in INVALID_FONTS:
        if name.lower() in str(filename).lower():
            return False
    return True
