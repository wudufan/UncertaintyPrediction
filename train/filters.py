'''
additional filters for uncertainty
'''

import scipy.ndimage


def gaussian_filter_2d(y, std):
    y = scipy.ndimage.gaussian_filter1d(y, std, 1)
    y = scipy.ndimage.gaussian_filter1d(y, std, 2)

    return y
