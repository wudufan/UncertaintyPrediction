# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage
import SimpleITK as sitk
import imageio
import scipy.stats

# %%
# input_dir = '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/'\
#             'uncertainty/denoising_l2_depth_4/l2_depth_4/img/valid'
input_dir = '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/'\
            'uncertainty/nlm_0.05/l2_depth_4/img/valid'
# input_dir = '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/'\
#             'denoising/l2_depth_4/valid'
# input_dir = '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/'\
#             'uncertainty/denoising_l2_depth_4/l2_depth_4/filter_3/img/valid'
# input_dir = '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/'\
#             'gaussian_assumption/l2_depth_4/img/valid'
tag = 'dose_rate_4'
scale_y = 100
norm_x = 1
norm_p = scale_y * 100 / 1000
norm_y = scale_y * 100 / 1000

x = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, tag + '.x.nii'))).astype(np.float32) / norm_x
preds = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, tag + '.pred.nii'))).astype(np.float32) / norm_p
y = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, tag + '.y.nii'))).astype(np.float32) / norm_y

# y = np.abs(x - y)

# %%
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(np.sqrt(np.mean(preds[-50:]**2, 0)), 'gray', vmin=0, vmax=50)
plt.title('Mean predicted 50 slices (std) \n [0, 50] HU')
plt.subplot(122)
plt.imshow(np.sqrt(np.mean(y[-50:]**2, 0)), 'gray', vmin=0, vmax=50)
plt.title('Mean error 50 slices (std) \n [0, 50] HU')

# %%
islice = 80
std = 10
smooth_y = np.sqrt(scipy.ndimage.gaussian_filter(y[islice]**2, std))
smooth_p = np.sqrt(scipy.ndimage.gaussian_filter(preds[islice]**2, std))

plt.figure(figsize=(16, 4))
plt.subplot(141)
plt.imshow(preds[islice], 'gray', vmin=0, vmax=5)
plt.title('predicted')
plt.subplot(142)
plt.imshow(y[islice], 'gray', vmin=0, vmax=5)
plt.title('error')
plt.subplot(143)
plt.imshow(smooth_p, 'gray', vmin=0, vmax=5)
plt.title('smoothed predicted \n (Gaussian with std=10)')
plt.subplot(144)
plt.imshow(smooth_y, 'gray', vmin=0, vmax=5)

# %%
islice = 295
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(x[islice], 'gray', vmin=-160, vmax=240)
plt.title('x: [-160,240] HU')
plt.subplot(132)
plt.imshow(preds[islice], 'gray', vmin=0, vmax=50)
plt.title('uncertainty (std): [0, 50] HU')
plt.subplot(133)
plt.imshow(y[islice], 'gray', vmin=0, vmax=50)
plt.title('|y-x|: [0, 50] HU')

# %%
var_y_mean = np.mean((y)**2, (1, 2))
var_pred_mean = np.mean((preds)**2, (1, 2))

plt.figure()
plt.plot(var_y_mean)
plt.plot(var_pred_mean)
plt.ylabel('Mean Variance (HU$^2$)')
plt.xlabel('#slice')
plt.legend(['$(y-f(x))^2$', 'Predicted'])

# %%
# generate figures
output_dir = os.path.join('figure', tag)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# save slices
def save_img(filename, img, vmin, vmax):
    img = (img - vmin) / (vmax - vmin) * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    if filename is None:
        plt.figure()
        plt.imshow(img, 'gray')
    else:
        imageio.imwrite(filename, img)


save_img(os.path.join(output_dir, 'x_shoulder.png'), x[101][64:-64, 64:-64], -160, 240)
save_img(os.path.join(output_dir, 'pred_shoulder.png'), preds[101][64:-64, 64:-64], 0, 40)
# save_img(os.path.join(output_dir, 'y_shoulder.png'), y[101][64:-64, 64:-64], -160, 240)

save_img(os.path.join(output_dir, 'x_liver.png'), x[295][64:-64, 64:-64], -160, 240)
save_img(os.path.join(output_dir, 'pred_liver.png'), preds[295][64:-64, 64:-64], 0, 40)
# save_img(os.path.join(output_dir, 'y_liver.png'), y[295][64:-64, 64:-64], -160, 240)

# %%
# quantification
# interslice mean
y_mean = np.sqrt(np.mean(y[:100]**2, 0))
pred_mean = np.sqrt(np.mean(preds[:100]**2, 0))
save_img(os.path.join(output_dir, 'y_mean_100.png'), y_mean, vmin=0, vmax=40)
save_img(os.path.join(output_dir, 'pred_mean_100.png'), pred_mean, vmin=0, vmax=40)

rmse = np.sqrt(np.mean((y_mean - pred_mean)**2))
psnr = 20 * np.log10(pred_mean.max() / rmse)
print(rmse, psnr)

# %%
# whole-slice mean
y_slice_mean = np.sqrt(np.mean(y**2, (1, 2)))
pred_slice_mean = np.sqrt(np.mean(preds**2, (1, 2)))

plt.figure(figsize=[4, 3], dpi=200)
plt.plot(pred_slice_mean)
plt.plot(y_slice_mean, '--')
plt.xlabel('Slice index')
plt.ylabel('Mean error of each slice (HU)')
plt.legend(['Predicted', 'Ground truth'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'slice_mean_err.png'))

# relative error
errs = np.abs(pred_slice_mean - y_slice_mean) / y_slice_mean
print(errs.min(), errs.max(), np.median(errs))
print(errs.mean(), np.std(errs))

# r = scipy.stats.pearsonr(y_slice_mean, pred_slice_mean)
# print (r)

# p = np.polyfit(y_slice_mean, pred_slice_mean, 1)
# xmin = 7
# xmax = 27

# plt.figure(figsize = [3,3], dpi=200)
# plt.plot(y_slice_mean.flatten(), pred_slice_mean.flatten(), '.', markersize=5)
# plt.plot([xmin, xmax], [p[0]*xmin + p[1], p[0]*xmax+p[1]], '--')
# plt.plot([xmin, xmax], [xmin, xmax], 'k-')

# %%
