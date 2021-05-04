import numpy as np

PDF = np.ndarray
CDF = np.ndarray
CDFs = np.ndarray
PDFs = np.ndarray
MC_choices = np.ndarray


def monte_carlo_from_1d_cdf(cdf: CDF, n_samples: int = 1) -> MC_choices:
    """A Monte Carlo (MC) trial for given cumlative distribution
    function (CDF)

    Args:
        cdf: A 1-D ndarray with values being the CDF
        n_samples: the number of samples to draw from the CDF

    Returns:
        The result of the MC draw for each samples.

    Notes:
        The MC algo performs no checks on the CDFs
    """

    # sample MC distibution
    rand = np.random.uniform(size=n_samples)
    # gets the approptiate cdfs values
    mask = cdf > rand[:, None]
    return np.argmax(mask, axis=1)


def monte_carlo_from_1d_pdf(pdf: PDF, n_samples: int = 1) -> MC_choices:
    """A Monte Carlo (MC) trial for given probability distribution
    function (PDF)

    Args:
        pdf: A 1-D ndarray with values being the PDF
        n_samples: the number of samples to draw from the PDF

    Returns:
        The result of the MC draw for each samples.

    Notes:
        The MC algo performs no checks on the CDFs
    """
    return monte_carlo_from_1d_cdf(np.cumsum(pdf), n_samples=n_samples)


def monte_carlo_from_cdf(cdf_s: CDFs) -> MC_choices:
    """A Monte Carlo (MC) trial for given cumlative distribution
    functions (CDFs)

    Args:
        cdf_s: A 2-D ndarray with
            dimension 0 = number of sample,
            dimension 1 = size of the CDFs.

    Returns:
        The result of the MC draw for each samples.

    Notes:
        The MC algo performs no checks on the CDFs
    """
    # sample MC distibution
    rand = np.random.uniform(size=cdf_s.shape[0])
    # gets the approptiate CDFs values
    mask = cdf_s > rand[:, None]
    return np.argmax(mask, axis=1)


def monte_carlo_from_pdf(pdf_s: PDFs) -> MC_choices:
    """A Monte Carlo (MC) trial for given probability distribution
    functions (PDFs)

    Args:
        pdf_s: A 2-D ndarray with
            dimension 0 = number of sample,
            dimension 1 = size of the PDFs.

    Returns:
        The result of the MC draw for each samples.

    Notes:
        The MC algo performs no checks on the PDFs
    """
    return monte_carlo_from_cdf(np.cumsum(pdf_s, axis=1))


