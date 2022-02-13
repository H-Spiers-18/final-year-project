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
