import numpy as np
from scipy.stats import wilcoxon


def get_wilcoxon_p_value(s1, s2):
    """
    Performs the Wilcoxon test to determine whether the difference between the test and predicted values
    is statistically significant
    Parameters
    ----------
    s1: numpy.ndarray - array of prediction values supporting the null hypothesis
    s2: numpy.ndarray - array of prediction values supporting the alternative hypothesis
    Returns
    -------
    p: float - p-value used to quantify the difference between y_test and y_pred. if the value is <= 0.05 then
               we can consider the difference to be statistically significant
    """
    _, p = wilcoxon(s1, s2)
    return p


def get_cliffs_delta(s1, s2):
    """
    Performs the Cliff's delta test to determine the effect size of my measurements. Based on the following link's
    implementation, just written to efficiently use numpy arrays - https://github.com/txt/ase16/blob/master/doc/stats.md
    Parameters
    ----------
    s1: numpy.ndarray - array of prediction values supporting the null hypothesis
    s2: numpy.ndarray - array of prediction values supporting the alternative hypothesis
    Returns
    -------
    z: float - z-value used to determine effect size.
               z < 0.147 - negligible effect size
               0.147 <= z < 0.33 - small effect size,
               0.33 <= z < 0.474 - medium effect size,
               0.474 <= z - large effect size
               (see figure 11 in report, based on D. Chen et. al [31])
    """
    # ensure that the 2 samples are numpy arrays
    if type(s1) != np.ndarray:
        s1 = np.array(s1)
    if type(s2) != np.ndarray:
        s2 = np.array(s2)

    less_than_count = greater_than_count = 0
    for x in s1:
        greater_than_count += np.sum(s2 > x)
        less_than_count += np.sum(s2 < x)

    z = abs(less_than_count - greater_than_count) / (len(s1) * len(s2))
    return z
