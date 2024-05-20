#!/usr/bin/env python
# coding: utf-8

# # SETUP

# In[ ]:


from IPython.display import clear_output

get_ipython().system('pip install -r /kaggle/input/check-image-orientation/requirements.txt')
get_ipython().system('pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps')
get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints')
get_ipython().system('cp /kaggle/input/aliked/pytorch/aliked-n16/1/* /root/.cache/torch/hub/checkpoints/')
get_ipython().system('cp /kaggle/input/lightglue/pytorch/aliked/1/* /root/.cache/torch/hub/checkpoints/')
get_ipython().system('cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth')
get_ipython().system('cp /kaggle/input/check-image-orientation/2020-11-16_resnext50_32x4d.zip /root/.cache/torch/hub/checkpoints/')

clear_output(wait=False)


# In[ ]:


from pathlib import Path
from copy import deepcopy
import numpy as np
import math
import pandas as pd
import pandas.api.types
from itertools import combinations
import sys, torch, h5py, pycolmap, datetime
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia as K
import kornia.feature as KF
from lightglue.utils import load_image
from lightglue import LightGlue, ALIKED, match_pair
from transformers import AutoImageProcessor, AutoModel
from check_orientation.pre_trained_models import create_model
sys.path.append("/kaggle/input/colmap-db-import")
from database import *
from h5_to_db import *

IMC_PATH = '/kaggle/input/image-matching-challenge-2024'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clear_output(wait=False)


# # CHECK IMAGE ORIENTATION

# In[ ]:


def rotate_image(image, rotation):
    for i in range(4):
        with torch.no_grad():
            pred = rotation(image[None, ...]).argmax()
        if pred == 0: break
        image = image.rot90(dims=[1, 2])
    return image


# # OVERLAP DETECTION

# In[ ]:


def overlap_detection(extractor, matcher, image0, image1, min_matches):
    feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
    if len(matches01['matches']) < min_matches:
        return feats0, feats1, matches01
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    left0, top0 = m_kpts0.numpy().min(axis=0).astype(int)
    width0, height0 = m_kpts0.numpy().max(axis=0).astype(int)
    height0 -= top0
    width0 -= left0
    left1, top1 = m_kpts1.numpy().min(axis=0).astype(int)
    width1, height1 = m_kpts1.numpy().max(axis=0).astype(int)
    height1 -= top1
    width1 -= left1
    crop_box0 = (top0, left0, height0, width0)
    crop_box1 = (top1, left1, height1, width1)
    cropped_img_tensor0 = TF.crop(image0, *crop_box0)
    cropped_img_tensor1 = TF.crop(image1, *crop_box1)
    feats0_c, feats1_c, matches01_c = match_pair(extractor, matcher, cropped_img_tensor0, cropped_img_tensor1)
    feats0_c['keypoints'][..., 0] += left0
    feats0_c['keypoints'][..., 1] += top0
    feats1_c['keypoints'][..., 0] += left1
    feats1_c['keypoints'][..., 1] += top1
    return feats0_c, feats1_c, matches01_c


# # SUBMISSION

# In[ ]:


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def parse_sample_submission(data_path):
    data_dict = {}
    with open(data_path, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                print("header:", l)

            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(',')
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(IMC_PATH + '/' + image_path))

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    return data_dict

def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


# In[ ]:


def create_submission(results, data_dict, base_path):    
    with open("submission.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        
        for dataset in data_dict:
            if dataset in results:
                res = results[dataset]
            else:
                res = {}
            
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}
                    
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    image_path = str(image.relative_to(base_path))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")


# In[ ]:


def run(data_path, get_pairs, keypoints_matches, ransac_and_sparse_reconstruction, submit=True):
    results = {}
    
    data_dict = parse_sample_submission(data_path)
    datasets = list(data_dict.keys())
    
    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}
            
        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]

            index_pairs = get_pairs(image_paths)
            keypoints_matches(image_paths, index_pairs)                
            maps = ransac_and_sparse_reconstruction(image_paths[0].parent)
            clear_output(wait=False)
            
            path = 'test' if submit else 'train'
            images_registered  = 0
            best_idx = 0
            for idx, rec in maps.items():
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx
                    
            for k, im in maps[best_idx].images.items():
                key = Path(IMC_PATH) / path / scene / "images" / im.name
                results[dataset][scene][key] = {}
                results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            create_submission(results, data_dict, Path(IMC_PATH))


# # mAA METRIC

# In[ ]:


_EPS = np.finfo(float).eps * 4.0

# mAA evaluation thresholds per scene, different accoring to the scene
translation_thresholds_meters_dict = {
 'multi-temporal-temple-baalshamin':  np.array([0.025,  0.05,  0.1,  0.2,  0.5,  1.0]),
 'pond':                              np.array([0.025,  0.05,  0.1,  0.2,  0.5,  1.0]),
 'transp_obj_glass_cylinder':         np.array([0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]),
 'transp_obj_glass_cup':              np.array([0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]),
 'church':                            np.array([0.025,  0.05,  0.1,  0.2,  0.5,  1.0]),
 'lizard':                            np.array([0.025,  0.05,  0.1,  0.2,  0.5,  1.0]),
 'dioscuri':                          np.array([0.025,  0.05,  0.1,  0.2,  0.5,  1.0]), 
}


def vector_norm(data, axis=None, out=None):
    '''Return length, i.e. Euclidean norm, of ndarray along axis.'''
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None


def quaternion_matrix(quaternion):
    '''Return homogeneous rotation matrix from quaternion.'''
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# based on the 3D registration from https://github.com/cgohlke/transformations
def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims: 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]

    return M


# This is the IMC 3D error metric code
def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    
    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))

    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n-2):
        for jj in range(ii+1, n-1):
            for kk in range(jj+1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(ev_coord[:, i], gt_coord[:, i], usesvd=False)
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord)**2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):                        
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]])
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord)**2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze((no_inl_ref > max_no_inl) | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)), axis=0)
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update])
                        best_transf_matrix[to_update] = transf_matrix

    best_model = {
        "valid_cams": idx_cams,        
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix}
    return best_model


# mAA computation
def mAA_on_cameras(err, thresholds, n, skip_top_thresholds, to_dec=3):
    
    aux = err[:, skip_top_thresholds:] < np.expand_dims(np.asarray(thresholds[skip_top_thresholds:]), axis=0)
    return np.sum(np.maximum(np.sum(aux, axis=0) - to_dec, 0)) / (len(thresholds[skip_top_thresholds:]) * (n - to_dec))


# import data - no error handling in case float(x) fails
def get_camera_centers_from_df(df):
    out = {}
    for row in df.iterrows():
        row = row[1]
        fname = row['image_path']
        R = np.array([float(x) for x in (row['rotation_matrix'].split(';'))]).reshape(3, 3)
        t = np.array([float(x) for x in (row['translation_vector'].split(';'))]).reshape(3)
        center = -R.T @ t
        out[fname] = center
    return out


def evaluate_rec(gt_df, user_df, inl_cf=0.8, strict_cf=0.5, skip_top_thresholds=2, to_dec=3,
                 thresholds=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]):
    # get camera centers
    ucameras = get_camera_centers_from_df(user_df)
    gcameras = get_camera_centers_from_df(gt_df)    

    # the denominator for mAA ratio
    m = gt_df.shape[0]
    
    # get the image list to use
    good_cams = []
    for image_path in gcameras.keys():
        if image_path in ucameras.keys():
            good_cams.append(image_path)
        
    # put corresponding camera centers into matrices
    n = len(good_cams)
    u_cameras = np.zeros((3, n))
    g_cameras = np.zeros((3, n))
    
    ii = 0
    for i in good_cams:
        u_cameras[:, ii] = ucameras[i]
        g_cameras[:, ii] = gcameras[i]
        ii += 1
        
    # Horn camera centers registration, a different best model for each camera threshold
    model = register_by_Horn(u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf)
    
    # transformation matrix
    T = np.squeeze(model['transf_matrix'][-1])
    
    # mAA
    mAA = mAA_on_cameras(model["err"], thresholds, m, skip_top_thresholds, to_dec)
    return mAA


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    
    scenes = list(set(solution['dataset'].tolist()))
    results_per_dataset = []
    for dataset in scenes:
        gt_ds = solution[solution['dataset'] == dataset]
        user_ds = submission[submission['dataset'] == dataset]
        gt_ds = gt_ds.sort_values(by=['image_path'], ascending=True)
        user_ds = user_ds.sort_values(by=['image_path'], ascending=True)
        result = evaluate_rec(gt_ds, user_ds, inl_cf=0, strict_cf=-1, skip_top_thresholds=0, to_dec=3,
                              thresholds=translation_thresholds_meters_dict[dataset])
        results_per_dataset.append(result)
    return float(np.array(results_per_dataset).mean())


# Define the additional functions
def process_image(img_path, processor, model, device):
    """Process a single image and return its embedding."""
    image = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt", do_rescale=False, do_resize=True, 
                           do_center_crop=True, size=224).to(device)
        outputs = model(**inputs)
        embedding = F.normalize(outputs.last_hidden_state.max(dim=1)[0])
    return embedding

def get_image_embeddings(images_list, processor, model, device):
    """Get embeddings for a list of images."""
    embeddings = []
    for img_path in images_list:
        embedding = process_image(img_path, processor, model, device)
        embeddings.append(embedding)
    return torch.cat(embeddings, dim=0)

def calculate_distance_matrix(embeddings, threshold, tolerance):
    """Calculate the distance matrix and create a boolean distance matrix."""
    distances = torch.cdist(embeddings, embeddings).cpu()
    distance_matrix = (distances <= threshold).numpy()
    np.fill_diagonal(distance_matrix, False)
    
    # Handle zero-distance cases
    zero_distance_indices = np.where(distance_matrix.sum(axis=1) == 0)[0]
    for idx in zero_distance_indices:
        closest_indices = np.argsort(distances[idx])[1:MIN_PAIRS]
        distance_matrix[idx, closest_indices] = True
    
    # Apply tolerance threshold
    out_of_tolerance_indices = np.where(distances >= tolerance)
    distance_matrix[out_of_tolerance_indices] = False
    
    return distance_matrix

def find_image_pairs(distance_matrix):
    """Find pairs of similar images using the boolean distance matrix."""
    pairs = set()
    num_images = distance_matrix.shape[0]
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if distance_matrix[i, j]:
                pairs.add((i, j))
    return list(pairs)

def get_pairs(images_list, device=DEVICE):
    if EXHAUSTIVE:
        return list(combinations(range(len(images_list)), 2))
    
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1/')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1/').eval().to(device)
    
    embeddings = get_image_embeddings(images_list, processor, model, device)
    distance_matrix = calculate_distance_matrix(embeddings, DISTANCES_THRESHOLD, TOLERANCE)
    pairs = find_image_pairs(distance_matrix)
    
    return pairs

def extract_keypoints_and_descriptors(image_path, extractor, device):
    """Extract keypoints and descriptors for a single image."""
    with torch.inference_mode():
        image = load_image(image_path).to(device)
        feats = extractor.extract(image)
    return feats["keypoints"].squeeze().cpu().numpy(), feats["descriptors"].squeeze().detach().cpu().numpy()

def save_keypoints_and_descriptors(images_list, extractor, device):
    """Save keypoints and descriptors for all images to HDF5 files."""
    with h5py.File("keypoints.h5", mode="w") as f_kp, h5py.File("descriptors.h5", mode="w") as f_desc:  
        for image_path in images_list:
            keypoints, descriptors = extract_keypoints_and_descriptors(image_path, extractor, device)
            f_kp[image_path.name] = keypoints
            f_desc[image_path.name] = descriptors

def match_keypoints(images_list, pairs, matcher, device):
    """Match keypoints between pairs of images and save matches to an HDF5 file."""
    with h5py.File("keypoints.h5", mode="r") as f_kp, h5py.File("descriptors.h5", mode="r") as f_desc, \
         h5py.File("matches.h5", mode="w") as f_matches:  
        for pair in pairs:
            key1, key2 = images_list[pair[0]].name, images_list[pair[1]].name
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                _, idxs = matcher(desc1, desc2, KF.laf_from_center_scale_ori(kp1[None]), KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs): 
                group = f_matches.require_group(key1)
            if len(idxs) >= MIN_MATCHES: 
                group.create_dataset(key2, data=idxs.detach().cpu().numpy())

def keypoints_matches(images_list, pairs):
    """Extract keypoints and descriptors, then match keypoints between pairs of images."""
    extractor = ALIKED(max_num_keypoints=MAX_NUM_KEYPOINTS, detection_threshold=DETECTION_THRESHOLD, resize=RESIZE_TO).eval().to(DEVICE)
    matcher = KF.LightGlueMatcher("aliked", {'width_confidence': -1, 'depth_confidence': -1, 'mp': True if 'cuda' in str(DEVICE) else False}).eval().to(DEVICE)
    rotation = create_model("swsl_resnext50_32x4d").eval().to(DEVICE)
    
    save_keypoints_and_descriptors(images_list, extractor, DEVICE)
    match_keypoints(images_list, pairs, matcher, DEVICE)

def create_database():
    """Create a new COLMAP database with a unique timestamp."""
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    db_name = f'colmap_{time_str}.db'
    db = COLMAPDatabase.connect(db_name)
    db.create_tables()
    return db, db_name

def add_keypoints_and_matches(db, images_path):
    """Add keypoints and matches to the COLMAP database."""
    fname_to_id = add_keypoints(db, '/kaggle/working/', images_path, '', 'simple-pinhole', False)
    add_matches(db, '/kaggle/working/', fname_to_id)
    db.commit()
    return fname_to_id

def perform_sparse_reconstruction(db_name, images_path):
    """Perform sparse reconstruction using COLMAP."""
    pycolmap.match_exhaustive(db_name, sift_options={'num_threads': 1})
    maps = pycolmap.incremental_mapping(
        database_path=db_name, 
        image_path=images_path,
        output_path='/kaggle/working/', 
        options=pycolmap.IncrementalPipelineOptions({'min_model_size': MIN_MODEL_SIZE, 'max_num_models': MAX_NUM_MODELS, 'num_threads': 1})
    )
    return maps

def ransac_and_sparse_reconstruction(images_path):
    """Main function to create a database, add keypoints and matches, and perform sparse reconstruction."""
    db, db_name = create_database()
    add_keypoints_and_matches(db, images_path)
    maps = perform_sparse_reconstruction(db_name, images_path)
    return maps

# Constants
EXHAUSTIVE = True
MIN_PAIRS = 50
DISTANCES_THRESHOLD = 0.3
TOLERANCE = 500

MAX_NUM_KEYPOINTS = 4096
RESIZE_TO = 1280
DETECTION_THRESHOLD = 0.005
MIN_MATCHES = 100

MIN_MODEL_SIZE = 5
MAX_NUM_MODELS = 3

N_SAMPLES = 50

SUBMISSION = True

# Functions for data handling
def create_image_path(row):
    """Create the image path for a row in the dataframe."""
    row['image_path'] = 'train/' + row['dataset'] + '/images/' + row['image_name']
    return row

def load_and_prepare_data(imc_path, n_samples):
    """Load the training data and prepare the image paths."""
    train_df = pd.read_csv(f'{imc_path}/train/train_labels.csv')
    train_df = train_df.apply(create_image_path, axis=1).drop_duplicates(subset=['image_path'])
    return train_df

def sample_images(train_df, n_samples):
    """Sample images from the training data."""
    grouped = train_df.groupby(['dataset', 'scene'])['image_path']
    image_paths = []

    for _, group in grouped:
        n = min(n_samples, len(group))
        sampled_group = group.sample(n, random_state=42).reset_index(drop=True)
        image_paths.extend(sampled_group)

    return image_paths

def prepare_dataframes(train_df, image_paths):
    """Prepare the ground truth and prediction dataframes."""
    gt_df = train_df[train_df.image_path.isin(image_paths)].reset_index(drop=True)
    pred_df = gt_df[['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector']]
    pred_df.to_csv('pred_df.csv', index=False)
    return gt_df, pred_df

def evaluate_submission(gt_df, pred_df_path):
    """Evaluate the submission and print the mean Average Accuracy."""
    pred_df = pd.read_csv(pred_df_path)
    mAA = round(score(gt_df, pred_df), 4)
    print('*** Total mean Average Accuracy ***')
    print(f"mAA: {mAA}")

# Main function for cross-validation pipeline
def main(imc_path, n_samples):
    """Main function to run the cross-validation pipeline."""
    train_df = load_and_prepare_data(imc_path, n_samples)
    image_paths = sample_images(train_df, n_samples)
    gt_df, pred_df = prepare_dataframes(train_df, image_paths)
    run('pred_df.csv', get_pairs, keypoints_matches, ransac_and_sparse_reconstruction, submit=False)
    evaluate_submission(gt_df, 'submission.csv')

if not SUBMISSION:
    main(IMC_PATH, N_SAMPLES)

# Function for submission pipeline
def run_submission(imc_path):
    """Run the submission pipeline."""
    data_path = imc_path + "/sample_submission.csv"
    run(data_path, get_pairs, keypoints_matches, ransac_and_sparse_reconstruction)

if SUBMISSION:
    run_submission(IMC_PATH)
