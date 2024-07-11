import os

import numpy as np
import mrcfile
import pandas as pd
from scipy.spatial.transform import Rotation
from lxml import etree
from scipy.linalg import norm
# import tensorflow as tf
# import starfile
from config import *
from glob import glob
import h5py
import re
import nrrd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.cm as cm

# ********************* Utility Functions *********************
def euler_to_rotation_matrix(u_vec):
    rotation_matrix = Rotation.from_euler('zxz', u_vec, degrees=True).as_matrix()
    return np.asarray(rotation_matrix)


def rotation_matrix_to_euler(rot_mtrx):
    euler = Rotation.from_matrix(rot_mtrx).as_euler('zxz', degrees=True)
    return euler


def quaternion_to_rotation_matrix(q_vec):
    rotation_matrix = Rotation.from_quat(q_vec).as_matrix()
    return np.asarray(rotation_matrix)


def rotation_matrix_to_quaternion(rot_mtrx):
    quaternions = Rotation.from_matrix(rot_mtrx).as_quat()
    return quaternions


def write_xml(objlist, output_path):

    objl_xml = etree.Element('objlist')
    for i in range(len(objlist)):
        tidx = objlist['tomo_idx'][i]
        objid = objlist['obj_id'][i]
        lbl = objlist['label'][i]
        x = objlist['x'][i]
        y = objlist['y'][i]
        z = objlist['z'][i]
        phi = objlist['phi'][i]
        psi = objlist['psi'][i]
        the = objlist['the'][i]

        obj = etree.SubElement(objl_xml, 'object')

        if tidx is not None:
            obj.set('tomo_idx', str(tidx))
        if objid is not None:
            obj.set('obj_id', str(objid))

        obj.set('class_label', str(lbl))
        obj.set('x', '%.3f' % x)
        obj.set('y', '%.3f' % y)
        obj.set('z', '%.3f' % z)
        obj.set('phi', '%.3f' % phi)
        obj.set('psi', '%.3f' % psi)
        obj.set('the', '%.3f' % the)

    tree = etree.ElementTree(objl_xml)
    tree.write(output_path, pretty_print=True)


def read_xml(filename, data_type):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    for p in range(len(objl_xml)):
        obj = {'tomo_idx': int(objl_xml[p].get('tomo_idx')),
               'obj_id': int(objl_xml[p].get('obj_id')),
               'label': int(objl_xml[p].get('class_label')),
               'x': float(objl_xml[p].get('x')),
               'y': float(objl_xml[p].get('y')),
               'z': float(objl_xml[p].get('z'))}
        if data_type == "real":
            obj['phi'] = float(objl_xml[p].get('phi'))
            obj['psi'] = float(objl_xml[p].get('psi'))
            obj['theta'] = float(objl_xml[p].get('the'))
        elif data_type == "polnet":
            obj['q1'] = float(objl_xml[p].get('Q1'))
            obj['q2'] = float(objl_xml[p].get('Q2'))
            obj['q3'] = float(objl_xml[p].get('Q3'))
            obj['q4'] = float(objl_xml[p].get('Q4'))
        obj_list.append(obj)
    return obj_list


def write_mrc(tomo, fname, v_size=1, dtype=None, no_saxes=True):
    """
    Saves a tomo (3D dataset) as MRC file

    :param tomo: tomo to save as ndarray
    :param fname: output file path
    :param v_size: voxel size (default 1)
    :param dtype: data type (default None, then the dtype of tomo is considered)
    :param no_saxes: if True (default) then X and Y axes are swaped to cancel the swaping made by mrcfile package
    :return:
    """
    with mrcfile.new(fname, overwrite=True) as mrc:
        if dtype is None:
            if no_saxes:
                mrc.set_data(np.swapaxes(tomo, 0, 2))
            else:
                mrc.set_data(tomo)
        else:
            if no_saxes:
                mrc.set_data(np.swapaxes(tomo, 0, 2).astype(dtype))
            else:
                mrc.set_data(tomo.astype(dtype))
        mrc.voxel_size.flags.writeable = True
        mrc.voxel_size = (v_size, v_size, v_size)
        mrc.set_volume()


# reading MRC files
def read_mrc(filename, vox_size):
    """ reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """
    if type(vox_size) is not tuple:
        vox_size = (vox_size, vox_size, vox_size)
    with mrcfile.open(filename, mode='r+', permissive=True) as mc:
        mc.voxel_size = (vox_size[0], vox_size[1], vox_size[2])
        mrc_tomo = mc.data
    return mrc_tomo


def get_patch_center(tomo_dim, p_in, coordinates, shiftr):
    """
    function to modify particle position w.r.t borders or the amount of requested shift
    """
    x = int(coordinates[0])  # int(obj['x'])
    y = int(coordinates[1])  # int(obj['y'])
    z = int(coordinates[2])  # int(obj['z'])

    # Add random shift to coordinates:
    x_shift = np.random.choice(range(-shiftr, shiftr + 1))
    y_shift = np.random.choice(range(-shiftr, shiftr + 1))
    z_shift = np.random.choice(range(-shiftr, shiftr + 1))

    x = x + x_shift
    y = y + y_shift
    z = z + z_shift

    # Shift position if passes the borders:
    if x < p_in:
        x = p_in
    if y < p_in:
        y = p_in
    if z < p_in:
        z = p_in

    if x > tomo_dim[2] - p_in:
        x = tomo_dim[2] - p_in
    if y > tomo_dim[1] - p_in:
        y = tomo_dim[1] - p_in
    if z > tomo_dim[0] - p_in:
        z = tomo_dim[0] - p_in

    return x, y, z, x_shift, y_shift, z_shift


def normalize_eulers(real_labels):
    # Normalizing phi and theta to (-180, 180)
    # Normalizing Psi to (-90, 90)
    new_min_phi_theta, new_max_phi_theta = -180, 180
    new_min_psi, new_max_psi = -90, 90

    real_phi = [real_labels[i]['phi'] for i, val in enumerate(real_labels)]
    real_psi = [real_labels[i]['psi'] for i, val in enumerate(real_labels)]
    real_theta = [real_labels[i]['theta'] for i, val in enumerate(real_labels)]

    real_phi_min, real_phi_max = np.min(real_phi), np.max(real_phi)
    real_psi_min, real_psi_max = np.min(real_psi), np.max(real_psi)
    real_theta_min, real_theta_max = np.min(real_theta), np.max(real_theta)

    for i in range(len(real_phi)):
        real_phi[i] = ((real_phi[i] - real_phi_min) / (real_phi_max - real_phi_min))
        real_phi[i] *= (new_max_phi_theta - new_min_phi_theta)
        real_phi[i] += new_min_phi_theta
        real_labels[i]['phi'] = real_phi[i]

        real_psi[i] = ((real_psi[i] - real_psi_min) / (real_psi_max - real_psi_min))
        real_psi[i] *= (new_max_psi - new_min_psi)
        real_psi[i] += new_min_psi
        real_labels[i]['psi'] = real_psi[i]

        real_theta[i] = ((real_theta[i] - real_theta_min) / (real_theta_max - real_theta_min))
        real_theta[i] *= (new_max_phi_theta - new_min_phi_theta)
        real_theta[i] += new_min_phi_theta
        real_labels[i]['theta'] = real_theta[i]
    return real_labels


def data_augmentation(mol_3d_struct, rot_mtrx_label):
    """
    Rotates a patch 180 degree by two times 90-degree rotation
    """
    mol_3d_struct_90 = np.rot90(mol_3d_struct)
    mol_3d_struct_180 = np.rot90(mol_3d_struct_90)
    rot_mtrx_label_90 = np.rot90(rot_mtrx_label)
    rot_mtrx_label_180 = np.rot90(rot_mtrx_label_90)

    return mol_3d_struct_180, rot_mtrx_label_180

def cov_mtrx(data_dist):
    """
    C_S = 1/(n_s-1)((D_S.T * D_S) - ((1/n_s)*(1.T*D_S).T * (1.T*D_S)))
    Here 1 is a column vector with all elements equal to 1
    """

    ns = data_dist.shape[0]  # number of samples in distribution
    one = np.ones((data_dist.shape[0], 1))
    first_term = np.matmul(data_dist.T, data_dist)

    conj_mtrx = np.multiply(one.T, data_dist)
    second_term = np.multiply(conj_mtrx.T, conj_mtrx)

    c = (1/ns-1) * first_term - (1/ns) * second_term
    return c

def coral_loss(source_data, target_data):
    """
    paper: Deep CORAL - Correlation Alignment for Deep Domain Adaptation
    L_CORAL = 1/4*d^2*(Frobenious_norm(C_s-C_T)) --> what is d?
    """

    cs = cov_mtrx(source_data)
    ct = cov_mtrx(target_data)

    err = np.linalg.norm(cs, 'fro', axis=(1,2)) - np.linalg.norm(ct, 'fro', axis=(1, 2))
    print("CORAL error: ", err)

    return err


def gram_schmit(vec_6dof):
    # [N(a1)  N(a2-(b1.a2)b1) b1*b2]
    # N(a) = a / ||a||
    # a1: first column of 6DOF
    # a2: second column of 6DOF
    # b1, b2, b3 : columns of rotation matrix

    a1 = vec_6dof[0:3]
    a2 = vec_6dof[3:6]

    b1 = a1 / norm(a1)

    numerator2 = a2 - np.multiply(np.dot(b1, a2), b1)
    denominator2 = norm(numerator2)
    b2 = numerator2 / denominator2

    # b3 = np.multiply(b1, b2)
    e1 = [1., 0., 0.]
    e2 = [0., 1., 0.]
    e3 = [0., 0., 1.]

    b31 = np.linalg.det(np.transpose(np.vstack((b1, b2, e1))))
    b32 = np.linalg.det(np.transpose(np.vstack((b1, b2, e2))))
    b33 = np.linalg.det(np.transpose(np.vstack((b1, b2, e3))))

    b3 = [b31, b32, b33]
    rot_mtrx = np.transpose(np.vstack((b1, b2, b3)))
    return rot_mtrx


def xml_to_csv(file_dir_name, output_dir_name):
    compact_labels = read_xml(file_dir_name, config_dict['data_type'])

    if config_dict['data_type'] == "real":
        compact_labels = normalize_eulers(compact_labels)
    data_array = np.zeros((len(compact_labels), len(compact_labels[0].items())))

    columns_list = list(compact_labels[0].keys())
    df = pd.DataFrame(data_array, columns=columns_list)

    for key, val in enumerate(compact_labels):
        df.iloc[key][:] = list(list(compact_labels[key].values()))

    df.to_csv(output_dir_name, index=False)
    # file_dir_name -> os.path.join(data_dir, f"meta_info.xml")
    # output_dir_name -> os.path.join(data_dir, "real_meta_info.csv")


def txt2csv(txt_dir, out_dir):
    txt_files_list = list_files(txt_dir, 'txt')
    for txti in txt_files_list:
        df = pd.read_fwf(f'{txti}')
        csv_name = txti.split('/')[9].split(".")[0]
        df.to_csv(os.path.join(out_dir, f'{csv_name}.csv'), index=False)


def nrrd2csv(nrrd_file, out_dir):
    quaternion_labels, header = nrrd.read(nrrd_file)
    quaternion_labels = np.transpose(quaternion_labels)

    df = pd.DataFrame(quaternion_labels, columns=["q1", "q2", "q3", "q4"])
    df.to_csv(os.path.join(out_dir, 'nrrd.csv'), index=False)


def csv2nrrd(csv_file, out_dir):
    from collections import OrderedDict
    quaternion_labels = pd.read_csv(csv_file)
    quaternion_labels = quaternion_labels[["q1", "q2", "q3", "q4"]]
    header = OrderedDict([('type', 'double'), ('dimension', 2), ('sizes', np.array([26703,   4])), ('endian', 'little'), ('encoding', 'gzip')])
    quaternion_labels = np.transpose(np.array(quaternion_labels.values, dtype=np.float32))
    print(quaternion_labels.shape)
    print(quaternion_labels[:, 0:3])
    print(header)

    nrrd.write(os.path.join(out_dir, 'quaternion_labels.nrrd'), quaternion_labels, header)
    quat_labels, header = nrrd.read(os.path.join(out_dir, 'quaternion_labels.nrrd'))
    print(quat_labels.shape)
    print(quat_labels[:, 0:3])
    print(header)


def concat_meta_info(csv_dir):
    csv_files_list = list_files(csv_dir, 'csv')
    df = pd.read_csv(f'{csv_files_list[0]}')
    df["tomo"] = 0
    df = df.values
    cnt = 1
    for csvi in range(1, len(csv_files_list)):
        print(csvi)
        csv_df = pd.read_csv(f'{csv_files_list[csvi]}')
        csv_df["tomo"] = cnt
        csv_df = csv_df.values
        df = np.vstack((df, csv_df))
        cnt += 1
    df = pd.DataFrame(df, columns=["class", "z", "y", "x", "phi", "psi", "the", "tomo_idx"])
    df.to_csv(os.path.join(csv_dir, f"{config_dict['data_type']}_meta_info.csv"), index=False)


def geodesic(vec1, vec2):
    # This is used at the test time
    rot_mtrx1 = gram_schmit(vec1)
    rot_mtrx2 = gram_schmit(vec2)

    rot_mtrx = np.multiply(rot_mtrx1, np.linalg.inv(rot_mtrx2))
    geodesic_error = 1 / np.cos((np.trace(rot_mtrx) - 1) / 2)

    return geodesic_error


def write_txt(fdir, fname, fdata):
    with open(os.path.join(fdir, fname), "w") as text_file:
        text_file.write(fdata)


def sample_per_tomo(meta_file):
    cnt_tomo_samples = dict()
    df = pd.read_csv(meta_file)
    unique_tomo_idx = df.drop_duplicates(subset=['tomo_idx'])
    unique_tomo_idx = list(unique_tomo_idx['tomo_idx'])
    for i in range(len(unique_tomo_idx)):
        tomo_meta_info = df[df['tomo_idx'] == int(unique_tomo_idx[i])]
        cnt_tomo_samples[f"tomo_{int(unique_tomo_idx[i])}"] = tomo_meta_info.shape[0]
    return cnt_tomo_samples


def gen_gauss_noise(vol_dim):
    batch, row, col, depth, ch = vol_dim
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (batch, row, col, depth, ch))
    gauss_noise_vol = gauss.reshape(batch, row, col, depth, ch)
    return gauss_noise_vol


def list_directories(apath):
    list_directories = glob(apath, recursive=True)
    list_directories.sort(key=lambda f: int(re.sub('\D', '', f)))
    return list_directories


def list_files(dir_path, extension):
    list_files = glob(os.path.join(dir_path, f"*.{extension}"))
    list_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(f"List of Files: {list_files}")
    return list_files


def calc_residuals_mean(y_true, y_pred):

    rows = 2
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
    cnt = 0
    colors = cm.tab20b(np.linspace(0, 1, len(y_true)))
    annotatations = list([str(ss) for ss in range(1, len(y_true) + 1)])
    x = list(range(0, y_true.shape[0]))
    for r in range(0, rows):
        for c in range(0, cols):
            ax[r][c].scatter(x, y_true[:, cnt], c="b", s=15)
            for i, txt in enumerate(annotatations):
                ax[r][c].annotate(txt, (x[i], y_true[i, cnt]))
            ax[r][c].scatter(x, y_pred[:, cnt], c="r", s=15)
            for i, txt in enumerate(annotatations):
                ax[r][c].annotate(txt, (x[i], y_pred[i, cnt]))
            cnt += 1
    plt.show()


def count_true_positives(y_true, y_pred, representation_type, mole_type, mode, output_dir):
    """
    y_true:  tomo index + ground truth labels (it can be a 1+ n x 9, n x 6, n x 3, or n x 4 based on representation type)
    y_pred:  predicted labels (it can be a n x 9, n x 6, n x 3, or n x 4 based on representation type)
    representation type: can be 6dof, quat, euler, rot
    """
    degree_threshold = 20
    if mole_type == "4v4r":
        criterion = 0.038
    elif mole_type == "3j9i":
        criterion = 0.066
    tomo_idx = y_true[:, 0]
    true_predict = 0
    false_predict = 0
    err_degree = []
    outliers = []
    outlier_index = 0
    prev_tomo_idx = -1
    for i in range(len(y_true)):
        if representation_type == "euler":
            rot_mtrx1 = euler_to_rotation_matrix(y_true[i])
            rot_mtrx2 = euler_to_rotation_matrix(y_pred[i])
        elif representation_type == "quaternions":
            rot_mtrx1 = quaternion_to_rotation_matrix(y_true[i][4:8])
            rot_mtrx2 = quaternion_to_rotation_matrix(y_pred[i][4:8])
        elif representation_type == "6dof":
            rot_mtrx1 = gram_schmit(y_true[i])
            rot_mtrx2 = gram_schmit(y_pred[i])
        elif representation_type == "rot":
            rot_mtrx1 = np.reshape(y_true[i], (3, 3))
            rot_mtrx2 = np.reshape(y_pred[i], (3, 3))

        # I believe this is D' BEST formula to calculate angle difference between two rotation matrix
        geodesic_error = np.degrees(np.arccos(np.sum(np.multiply(rot_mtrx1, rot_mtrx2)) /
                                              (np.linalg.norm(rot_mtrx1) * np.linalg.norm(rot_mtrx2))))

        step_threshold = (degree_threshold / criterion)
        num_steps = int(geodesic_error / criterion)

        if num_steps < step_threshold:
            true_predict += 1
        else:
            if tomo_idx[i] != prev_tomo_idx:
                outlier_index = 0
                prev_tomo_idx = tomo_idx[i]

            false_predict += 1
        outlier_index += 1
        err_degree.append(geodesic_error)

    print(np.min(err_degree), np.max(err_degree))
    pd.DataFrame(err_degree).to_csv(os.path.join(output_dir, f"{mode}_angles_{representation_type}_{mole_type}.csv"),
                                    index=False, header=None)
    print(f"Number of correct predictions: {true_predict}")
    print(f"Number of False predictions: {false_predict}")


