import numpy as np


def apply_window_function(gt: np.ndarray, ref: np.ndarray,
                          window_size: int, measure_fn):
    image_shape = np.asarray(np.shape(gt))
    image_steps = np.floor(image_shape / window_size).astype(int)
    results = np.zeros(shape=image_steps)
    for x_idx in range(image_steps[0]):
        for y_idx in range(image_steps[1]):
            ref_window = ref[(x_idx*window_size):((x_idx+1)*window_size),
                             (y_idx*window_size):((y_idx+1)*window_size)]
            gt_window = gt[(x_idx*window_size):((x_idx+1)*window_size),
                           (y_idx*window_size):((y_idx+1)*window_size)]
            results[y_idx, x_idx] = measure_fn(ref_window, gt_window)

    return results
