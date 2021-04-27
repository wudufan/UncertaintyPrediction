'''
The noise model to generate x and y
'''

#%%
import numpy as np
import scipy.ndimage
import sklearn.preprocessing

#%%
class ImageNoiseModel:
    def __init__(self, 
                 x0: np.array,
                 var_roi_map: np.array = None,
                 var_roi_ratio = 0.33,
                 x0_blur_std = 10, 
                 std_x0_intercept = 0.01, 
                 std_x0_slope = 0.05, 
                 std_x_intercept = 0.01,
                 std_x_slope = 0.05, 
                 std_x_use_x0 = False):
        
        self.x0_blur_std = x0_blur_std
        self.std_x0_intercept = std_x0_intercept
        self.std_x0_slope = std_x0_slope
        self.std_x_intercept = std_x_intercept
        self.std_x_slope = std_x_slope
        self.std_x_use_x0 = std_x_use_x0

        self.x0 = np.copy(x0)
        if self.x0_blur_std > 0:
            self.x0_blur = scipy.ndimage.gaussian_filter(self.x0, self.x0_blur_std, mode = 'nearest')
        else:
            self.x0_blur = np.copy(self.x0)
        self.std_x0 =  self.std_x0_intercept + self.std_x0_slope * np.maximum(self.x0_blur, 0) 
        
        # variance roi provides excess variance for certain ROIs. The excess variance is independent for each pixel
        if var_roi_map is not None:
            self.var_roi_map = np.copy(var_roi_map)
        else:
            self.var_roi_map = np.zeros(self.x0.shape, np.uint8)
        self.var_roi_ratio = var_roi_ratio
    
    def forward_sample(self, n_sample_y:int = 1):
        x = np.random.normal(self.x0, self.std_x0)

        # additional variance
        one_hot_roi = sklearn.preprocessing.OneHotEncoder().fit_transform(self.var_roi_map.reshape([-1, 1])).toarray()
        one_hot_roi = one_hot_roi.reshape(list(self.var_roi_map.shape) + [-1])
        for i in range(1, one_hot_roi.shape[-1]):
            x += np.random.normal(0, self.std_x0) * one_hot_roi[..., i] * np.sqrt(i * self.var_roi_ratio)

        if n_sample_y <= 0:
            return x
        
        if self.std_x_use_x0:
            std_x = self.std_x_intercept + self.std_x_slope * np.maximum(self.x0, 0)
        else:
            std_x = self.std_x_intercept + self.std_x_slope * np.maximum(x, 0)
        y = []
        for i in range(n_sample_y):
            y.append(np.random.normal(x, std_x))
        
        return x, y

    def posterior_pdf(self, y, nsamples = 200, sample_width = 6):
        '''
        construct the posterior pdf p(x|y) for each pixel, from x0-sample_width*std_x0 to x0+sample_width*std_x0 with nsamples
        '''
        # construct the sampling point on x of the pdf
        pdf_x = []
        x0_flat = self.x0.flatten()

        # the std_x0 should consider the variance roi
        one_hot_roi = sklearn.preprocessing.OneHotEncoder().fit_transform(self.var_roi_map.reshape([-1, 1])).toarray()
        one_hot_roi = one_hot_roi.reshape(list(self.var_roi_map.shape) + [-1])
        std_x0 = np.copy(self.std_x0)
        for i in range(1, one_hot_roi.shape[-1]):
            std_x0 = std_x0 * np.sqrt(1 + i * self.var_roi_ratio * one_hot_roi[...,i])

        std_x0_flat = std_x0.flatten()
        for i in range(len(x0_flat)):
            dx = sample_width * 2 * std_x0_flat[i] / nsamples
            sx = x0_flat[i] - sample_width * std_x0_flat[i]
            pdf_x.append(np.arange(sx, sx + dx * nsamples - dx/2, dx))
        pdf_x = np.array(pdf_x).reshape(list(std_x0.shape) + [-1])

        # calculate the posterior probability
        if self.std_x_use_x0:
            std_x = self.std_x_intercept + self.std_x_slope * np.maximum(self.x0, 0)[..., np.newaxis]
        else:
            std_x = self.std_x_intercept + self.std_x_slope * np.maximum(pdf_x, 0)
        prob = 1 / std_x * np.exp(-(pdf_x - y[..., np.newaxis])**2 / (2 * std_x**2) - (pdf_x - self.x0[..., np.newaxis])**2 / (2 * std_x0[...,np.newaxis]**2))

        # prob
        prob = prob / np.sum(prob, -1)[..., np.newaxis]

        return pdf_x, prob
    
    def posterior_mean_and_std(self, y, nsamples = 200, sample_width = 6):
        pdf_x, prob = self.posterior_pdf(y, nsamples, sample_width)
        post_mean = np.sum(pdf_x * prob, -1)
        post_std = np.sqrt(np.sum(pdf_x**2 * prob, -1) - post_mean**2)

        return post_mean, post_std
        
    def posterior_mean_and_std_analytical(self, y):
        if not self.std_x_use_x0:
            raise ValueError('std_x_use_x0 must be true to calculate the mean and variance analytically')
        std_x = self.std_x_intercept + self.std_x_slope * np.maximum(self.x0, 0)
        one_hot_roi = sklearn.preprocessing.OneHotEncoder().fit_transform(self.var_roi_map.reshape([-1, 1])).toarray()
        one_hot_roi = one_hot_roi.reshape(list(self.var_roi_map.shape) + [-1])
        std_x0 = np.copy(self.std_x0)
        for i in range(1, one_hot_roi.shape[-1]):
            std_x0 = std_x0 * np.sqrt(1 + i * self.var_roi_ratio * one_hot_roi[...,i])

        var_x0 = std_x0**2
        var_x = std_x**2

        mean_val = (var_x0 * y + var_x * self.x0) / (var_x0 + var_x)
        std_val = np.sqrt((var_x0 * var_x) / (var_x0 + var_x))

        return mean_val, std_val

def sample1d(x0, y, x0_blur = None, std_x0_intercept = 0.01, std_x0_slope = 0.05, std_x_intercept = 0.01, std_x_slope = 0.1, use_x0_for_x = False, 
             nsamples = 200, sample_width = 6):
    if x0_blur is None:
        x0_blur = x0
    
    std_x0 = std_x0_intercept + std_x0_slope * np.maximum(x0, 0)
    dx = sample_width * 2 * std_x0 / nsamples
    sx = x0 - sample_width * std_x0
    pdf_x = np.arange(sx, sx + dx * nsamples - dx/2, dx)
    prob0 = np.exp(-(pdf_x - x0)**2 / (2 * std_x0**2))

    if use_x0_for_x:
        std_x = std_x_intercept + std_x_slope * np.maximum(x0, 0)
    else:
        std_x = std_x_intercept + std_x_slope * np.maximum(pdf_x, 0)
    prob = 1 / std_x * np.exp(-(pdf_x - y)**2 / (2 * std_x**2) - (pdf_x - x0)**2 / (2 * std_x0**2))

    return pdf_x, prob, prob0



#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    # comparing to theoretical sampling
    pdf_x, prob, prob0 = sample1d(1.072, 1.26, 0.22)
    p0 = prob0 / np.sum(prob0)
    px = prob / np.sum(prob)

    plt.figure()
    plt.plot(pdf_x, p0)
    plt.plot(pdf_x, px)
    plt.show()

    # the theoretical mean and variance
    # mean_th = (std0**2 * y + stdx**2 * x0) / (std0**2 + stdx**2)
    # var_th = (std0**2 * stdx**2) / (std0**2 + stdx**2)

    # the experimental mean and variance
    mean_exp = np.sum(px * pdf_x)
    var_exp = np.sum(px * pdf_x**2) - mean_exp**2

    # print (mean_th, var_th)
    print (mean_exp, var_exp)

    # # test noise model
    np.random.seed(100)
    x0 = sitk.GetArrayFromImage(sitk.ReadImage('./data/x0.nii')).astype(np.float32) / 1000 + 1
    var_roi = sitk.GetArrayFromImage(sitk.ReadImage('./data/variance.seg.nrrd'))[0]
    noise_model = ImageNoiseModel(x0, var_roi, std_x_use_x0=False)

    x, y = noise_model.forward_sample()

    plt.figure(figsize = [15, 5])
    plt.subplot(131); plt.imshow(x0[128:-128, 128:-128], 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(132); plt.imshow(x[128:-128, 128:-128], 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(133); plt.imshow(y[0][128:-128, 128:-128], 'gray', vmin=0.84, vmax=1.24)

    # xs = []
    # for i in range(700):
    #     if (i+1) % 10 == 0:
    #         print (i+1, end=',')
    #     xs.append(noise_model.forward_sample(0))
    # xs = np.array(xs)
    # plt.figure()
    # plt.imshow(np.std(xs, 0)[128:-128, 128:-128], 'gray', vmin=0.01 + 0.84 * 0.05, vmax=0.01 + 1.24*0.05)

    # posterior
    post_mean, post_std = noise_model.posterior_mean_and_std(y[0])
    error_x0 = np.sqrt(post_std**2 + post_mean**2 - 2*post_mean*x0 + x0**2)
    # pdf_x, prob = noise_model.posterior_pdf(y[0])

    # post_mean = np.sum(pdf_x * prob, -1)
    # post_std = np.sqrt(np.sum(pdf_x**2 * prob, -1) - post_mean**2)

    plt.figure(figsize=[8,8])
    plt.subplot(221); plt.imshow(post_mean[128:-128, 128:-128], 'gray', vmin=0.84, vmax=1.24)
    plt.subplot(222); plt.imshow(error_x0[128:-128, 128:-128], 'gray', vmin=0, vmax=0.1)

    if noise_model.std_x_use_x0:
        th_mean, th_std = noise_model.posterior_mean_and_std_analytical(y[0])
        plt.subplot(223); plt.imshow(th_mean[128:-128, 128:-128], 'gray', vmin=0.84, vmax=1.24)
        plt.subplot(224); plt.imshow(th_std[128:-128, 128:-128], 'gray', vmin=0.035, vmax=0.05)



# %%
