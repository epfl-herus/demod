"""Various helpers for statistical distribution functions.

pdf = probability distribution function
cdf = cumulative distribution function

"""
import numpy as np


def rescale_pdf(pdf: np.ndarray, epsilon: float = 1e-1):
    """Rescale a pdf.

    Adjust the value of the pdf if there are roundoff errors, by rescaling
    all the values of the pdfs

    Parameters:
        pdf : ndarray, 2-dim
            a ndarray with dim0 being the number of sample and dim1 the pdfs of
            each samples
        epsilon : float
            the maximum error to allow correction

    Returns:
        ndarray, 2-dim(float) the corrected pdf
    """
    # first reduces the dimension of the given pdfs
    shape = np.shape(pdf)
    pdf = np.reshape(pdf, (-1, shape[-1]))

    if not np.all(
        np.logical_and(
            pdf.sum(axis=-1) < 1 + epsilon, pdf.sum(axis=-1) > 1 - epsilon
        )
    ):
        raise ValueError(
            "When computing CDF, the last elements of each cdfs are not"
            " all close enough to 1"
        )
    # rescale all over the difference with one
    return np.reshape(pdf / pdf.sum(axis=-1)[:, None], (shape))


def rescale_cdf(cdf: np.ndarray, epsilon: float = 1e-1):
    """Rescale a cdf.

    Adjust the value of the cdf if there are roundoff errors, by rescaling
    all the values of the cdfs

    Parameters:
        cdf: ndarray, 2-dim
            a ndarray with dim0 being the number of sample and dim1 the cdfs of
            each samples
        epsilon: the maximum error to allow correction

    Returns:
        ndarray, 2-dim(float), the corrected cdf
    """
    # first reduces the dimension of the given cdfs
    shape = np.shape(cdf)
    cdf = np.reshape(cdf, (-1, shape[-1]))

    if not np.all(
        np.logical_and(cdf[:, -1] < 1 + epsilon, cdf[:, -1] > 1 - epsilon)
    ):
        raise ValueError(
            "Last elements of each arrays in cdf are not close enough to 1"
        )
    # rescale all over the difference with one
    return np.reshape(cdf / cdf[:, -1][:, None], (shape))


def check_valid_cdf(cdf: np.ndarray, epsilon: float = 1e-6):
    """Check the validity of the given cdf.

    Check that the values are increasing.
    Check that it ends at 1.

    Parameters:
        cdf : ndarray, of any size, with last dimension being the cdfs

    Returns:
        True if the cdf is valid

    Raises:
        ValueError:
            if the cdf is invalid
    """
    # first reduces the dimension of the given cdfs
    shape = np.shape(cdf)
    cdf = np.reshape(cdf, (-1, shape[-1]))
    # checks that cdfs ends with value 1
    if not np.all(
        np.logical_and(cdf[:, -1] > 1.0 - epsilon, cdf[:, -1] < 1.0 + epsilon)
    ):
        print(cdf, cdf.shape)
        raise ValueError(
            "Last elements of each arrays in cdf are not all == 1"
        )
    # checks that elements of the cdfs are always decreasing
    if not np.all((cdf <= np.roll(cdf, -1, axis=1))[:, :-1]):
        raise ValueError("Some cdfs are decreasing")
    # checks that cdfs are always between 0 and 1
    if not np.all(np.logical_and(cdf <= 1 + epsilon, cdf >= 0)):
        raise ValueError("Some values in the cdf are not between 0 and 1")
    return True
