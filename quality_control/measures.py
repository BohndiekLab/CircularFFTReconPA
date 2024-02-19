import torch
import numpy as np
from sewar import rmse
from sklearn.metrics import mutual_info_score
from torchmetrics.image import StructuralSimilarityIndexMeasure, UniversalImageQualityIndex
from scipy.signal import correlate
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine


def validate_input(a, b):
    if len(a) != len(b):
        raise AssertionError("The number of wavelengths must be the same for both input samples")


def get_torch_tensor(np_array):
    """
    Takes a 2D or 3D numpy array representing a greyscale image and transforms it into a torch sensor for the
    purposes of computing an image quality measure.

    :param np_array: a 2D or 3D numpy array
    :return: a torch tensor of shape (1, zdim, xdim, ydim)
    """
    shape = np_array.shape
    if len(shape) == 2:
        sx, sy = shape
        sz = 1
    elif len(shape) == 3:
        sx, sy, sz = shape
    else:
        raise AssertionError("The input image must be 2D or 3D")

    return torch.from_numpy(np_array.reshape((1, sz, sx, sy)))


def StructuralSimilarityIndex(expected_result, reconstructed_image):

    if len(np.shape(expected_result)) == 3:
        ssim = 0
        for i in range(len(expected_result)):
            gt = get_torch_tensor(expected_result[i])
            reco = get_torch_tensor(reconstructed_image[i])
            ssim += StructuralSimilarityIndexMeasure()(gt, reco).item()

        return ssim / len(expected_result)

    gt = get_torch_tensor(expected_result)
    reco = get_torch_tensor(reconstructed_image)
    return StructuralSimilarityIndexMeasure()(gt, reco).item()



def UniversalQualityIndex(expected_result, reconstructed_image):
    gt = get_torch_tensor(expected_result)
    reco = get_torch_tensor(reconstructed_image)
    return 1 - UniversalImageQualityIndex()(gt, reco).item()


def RootMeanSquaredError(expected_result, reconstructed_image):
    return rmse(expected_result, reconstructed_image)


def MutualInformation(expected_result, reconstructed_image):
    def precompute(data):
        data = data.reshape((-1,))
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data * 256
        return data.astype(int)
    gt = precompute(expected_result)
    reco = precompute(reconstructed_image)
    res = mutual_info_score(gt, reco)
    if res == 0:
        return 1
    return 1 / res


def WassersteinDistance(data1, data2):
    validate_input(data1, data2)

    emd = 0
    for i in range(len(data1)):
        emd += wasserstein_distance(data1[i], data2[i])
    emd = emd / len(data1)
    return emd


def CosineDistance(data1, data2):
    validate_input(data1, data2)

    cos_distance = 0
    for i in range(len(data1)):
        cos_distance += cosine(data1[i], data2[i])
    cos_distance = cos_distance / len(data1)
    return cos_distance


def NormalisedCrossCorrelation(data1, data2):
    print(np.shape(data1))
    print(np.shape(data2))
    # Ensure both input images are of the same size
    validate_input(data1, data2)

    ncc = 0
    for i in range(len(data1)):
        a = data1[i]
        b = data2[i]

        # Compute the mean of each image
        mean1 = np.mean(a)
        mean2 = np.mean(b)

        # Compute the cross-correlation using correlate2d
        cross_correlation = correlate(a - mean1, b - mean2, mode='full')

        # Compute the standard deviations of both images
        std1 = np.std(a)
        std2 = np.std(b)
        # Compute the NCC
        ncc += cross_correlation / (std1 * std2 * a.size)
    ncc = ncc / len(data1)

    return np.mean(ncc)


def BhattacharyyaDistance(data1, data2, range_min=-3, range_max=3, num_bins=61):

    bd = 0
    for i in range(len(data1)):
        a = data1[i]
        b = data2[i]
        # Z-score normalization
        mean1, std1 = np.mean(a), np.std(a)
        mean2, std2 = np.mean(b), np.std(b)

        zscore_normalized_image1 = (a - mean1) / std1
        zscore_normalized_image2 = (b - mean2) / std2

        # Create histograms
        hist1, bin_edges1 = np.histogram(zscore_normalized_image1, bins=num_bins, range=(range_min, range_max))
        hist2, bin_edges2 = np.histogram(zscore_normalized_image2, bins=num_bins, range=(range_min, range_max))

        # Calculate the Bhattacharyya distance
        hist1 = hist1 / np.sum(hist1)  # Normalize histograms
        hist2 = hist2 / np.sum(hist2)

        bd += -np.log(np.sum(np.sqrt(hist1 * hist2)))

    bd = bd / len(data1)

    return bd


def KullbackLeiblerDivergence(a, b):
    validate_input(a, b)
    # Normalise the data
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)

    # Compute discrete KLD from marginal histograms
    kld = 0
    for wl_idx in range(len(a)):
        marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_p = marginal_p + 0.00001
        marginal_q = marginal_q + 0.00001
        kld += entropy(marginal_p, marginal_q, base=2)
    kld = kld / len(a)
    return kld


def JensenShannonDivergence(a, b):
    validate_input(a, b)
    # Normalise the data
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)

    # Compute discrete JSD from marginal histograms
    jsd = 0
    for wl_idx in range(len(a)):
        marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
        marginal_p = marginal_p + 0.00001
        marginal_q = marginal_q + 0.00001
        jsd += jensenshannon(marginal_p, marginal_q, base=2)

    jsd = jsd / len(a)
    return jsd
