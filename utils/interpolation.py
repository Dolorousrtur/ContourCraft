import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def make_slerp(pA: np.ndarray, pB: np.ndarray, n_steps: int, omit_last: bool = True, duplicate_first=True) -> np.ndarray:
    """
    Create a slerp path between two rotations

    :param pA: [3,] first rotation vector
    :param pB: [3,] second rotation vector
    :param n_steps: number of steps between the two rotations
    :param omit_last: whether to omit the last rotation vector
    :return: [n_steps+1, 3] or [n_steps+2, 3] interpolated rotation vectors
    """


    times = np.linspace(0.0, 1.0, n_steps)
    if omit_last:
        times = times[:-1]

    if duplicate_first:
        times = np.concatenate([times[:1], times], axis=0)

    p = np.stack([pA, pB])
    p = R.from_rotvec(p)
    slerp = Slerp([0., 1.], p)
    interp = slerp(times).as_rotvec()

    return interp


def make_slerp_batch(pA: np.ndarray, pB: np.ndarray, n_steps: int, omit_last: bool = True):
    """
    Create a slerp path between two batches of rotation vectors

    :param pA: [Bx3] first batch of rotation vectors
    :param pB: [Bx3] second batch of rotation vectors
    :param n_steps: number of steps between the two rotations
    :param omit_last: whether to omit the last rotation vector

    :return: [n_steps+1, B, 3] or [n_steps+2, B, 3] interpolated rotation vectors
    """
    slerped = []
    B = pA.shape[0]

    for i in range(B):
        s = make_slerp(pA[i], pB[i], n_steps, omit_last=omit_last)
        slerped.append(s)
    slerped = np.stack(slerped, axis=1)

    return slerped


def prepend_interpolation_linear(array_sequence, n_frames, start_array=None, duplicate_first=True):
    """
    Prepends a sequence with linear interpolation from `start_array` or zeros 
    to the first frame of the sequence. 

    :param array_sequence: np.array of shape [n_frames, K]
    :param n_frames: number of frames to prepend
    :param start_array: np.array of shape [K] to interpolate from, if None, zeros are used
    """

    if start_array is None:
        start_array = np.zeros_like(array_sequence[0])

    start_array = start_array[None, :]
    end_interpolation = array_sequence[0][None, :]

    t_vals = np.linspace(0, 1, n_frames)[:-1, None]
    if duplicate_first:
        t_vals = np.concatenate([t_vals[:1], t_vals], axis=0)

    inter_values = start_array + t_vals * (end_interpolation - start_array)

    out_sequence = np.concatenate([inter_values, array_sequence], axis=0)

    return out_sequence


def prepend_interpolation_slerp(array_sequence, n_frames, start_array=None, duplicate_first=True):
    """
    Prepends a sequence with spherical linear interpolation from `start_array` or zeros 
    to the first frame of the sequence. 

    :param array_sequence: np.array of shape [n_frames, K], where K//3 == 0
    :param n_frames: number of frames to prepend
    :param start_array: np.array of shape [K] to interpolate from, if None, zeros are used
    """

    if start_array is None:
        start_array = np.zeros_like(array_sequence[0])

    start_array = start_array[None, :]
    end_interpolation = array_sequence[0][None, :]

    t_vals = np.linspace(0, 1, n_frames)[:-1, None]
    if duplicate_first:
        t_vals = np.concatenate([t_vals[:1], t_vals], axis=0)

    start_array = start_array.reshape(-1, 3)
    end_interpolation = end_interpolation.reshape(-1, 3)

    inter_values = make_slerp_batch(start_array, end_interpolation, n_frames)

    inter_values = inter_values.reshape(inter_values.shape[0], -1)

    out_sequence = np.concatenate([inter_values, array_sequence], axis=0)

    return out_sequence

def repeat_first_frame(array_sequence, n_frames):
    """
    Repeats the first frame of the sequence `n_frames` times

    :param array_sequence: np.array of shape [n_frames, K]
    :param n_frames: number of frames to repeat
    """

    start_array = array_sequence[0][None, :]
    prepend_array = np.concatenate([start_array] * n_frames, axis=0)

    out_sequence = np.concatenate([prepend_array, array_sequence], axis=0)

    return out_sequence