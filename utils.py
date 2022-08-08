import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from skimage import measure


def make_dirs(path):
    """
    Creates the directory as specified from the path
    in case it exists it deletes it
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def resample(image, slice_thickness, pixel_spacing, new_spacing=None):
    # Determine current pixel spacing
    if new_spacing is None:
        new_spacing = [1, 1, 1]

    spacing = np.array([slice_thickness] + list(pixel_spacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)

    # recompute the resize factor and spacing such that we match the rounded new shape above
    rounded_resize_factor = new_shape / image.shape
    rounded_new_spacing = spacing / rounded_resize_factor

    # zoom with resize factor
    image = scipy.ndimage.interpolation.zoom(image, rounded_resize_factor, mode='nearest')

    return image, rounded_new_spacing


def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def plot_image(image, masked_image):
    fig, ax = plt.subplots(1, 2, figsize=[12, 12])
    ax[0].set_title('Original Slice')
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')

    ax[1].set_title('Masked Slice')
    ax[1].imshow(masked_image, cmap='gray')
    ax[1].axis('off')
    plt.show()


def create_vessel_mask(lung_mask, ct_numpy, level, window, denoise=False):
    vessels = lung_mask * ct_numpy  # isolate lung area
    vessels[vessels >= 0] = 0
    vessels[vessels < 0] = 1

    contours = intensity_seg(ct_numpy, level, window)
    lungs_contour = find_lungs(contours)

    if denoise:
        return denoise_vessels(lungs_contour, vessels)

    return vessels


def create_mask_from_polygon(image, contours):
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
        image: the image that the contours refer to
        contours: list of contours
    Returns:
    """

    lung_mask = np.array(Image.new('L', image.shape, 0))

    x = contours[0][:, 0]
    y = contours[0][:, 1]
    polygon_tuple = list(zip(x, y))
    img = Image.new('L', image.shape, 0)
    ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
    mask = np.array(img)
    lung_mask += mask

    if len(contours) > 1:
        x = contours[1][:, 0]
        y = contours[1][:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask -= mask

    lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary

    return lung_mask.T  # transpose it to be aligned with the image dims


def intensity_seg(ct_numpy, level=-200, window=200):
    min = level - window
    max = level + window
    clipped = clip_ct(ct_numpy, min, max)
    return measure.find_contours(clipped)


def set_is_closed(contour):
    if contour_distance(contour) < 1:
        return True
    else:
        return False


def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point
     to infer if the set is closed.
    Args:
        contour: np array of x and y points
    Returns: euclidean distance of first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    return euclidean_dist(dx, dy)


def euclidean_dist(dx, dy):
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


def find_lungs(contours):
    """
    Chooses the contours that correspond to the lungs and the body
    FIrst we exclude non closed sets-contours
    Then we assume some min area and volume to exclude small contours
    Then the body is excluded as the highest volume closed set
    The remaining areas correspond to the lungs
    Args:
        contours: all the detected contours
    Returns: contours that correspond to the lung area
    """
    body_and_lung_contours = []
    vol_contours = []

    for contour in contours:
        hull = ConvexHull(contour)

        if hull.volume > 2000 and set_is_closed(contour):
            body_and_lung_contours.append(contour)
            vol_contours.append(hull.volume)

    if len(body_and_lung_contours) == 2 or len(body_and_lung_contours) == 1:
        return body_and_lung_contours
    elif len(body_and_lung_contours) > 2:
        vol_contours, body_and_lung_contours = (list(t) for t in
                                                zip(*sorted(zip(vol_contours, body_and_lung_contours))))
        body_and_lung_contours.pop(-1)
        return body_and_lung_contours


def show_contour(image, contours, name=None, save=False):
    fig, ax = plt.subplots()
    ax.imshow(image.T, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(name)
        plt.close(fig)
    else:
        plt.show()


def set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width / 2)
    max_value = custom_center + (custom_width / 2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def overlay_plot(im, mask):
    plt.figure()
    plt.imshow(im.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)


def save_nifty(img_np, name, affine):
    """
    binary masks should be converted to 255 so it can be displayed in a nii viewer
    we pass the affine of the initial image to make sure it exits in the same
    image coordinate space
    Args:
        img_np: the binary mask
        name: output name
        affine: 4x4 np array
    Returns:
    """
    img_np[img_np == 1] = 255
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, name + '.nii.gz')


def clip_ct(ct_numpy, min_val, max_val):
    """
    Clips CT to predefined range and binarizes the values
    """
    clipped = ct_numpy.clip(min_val, max_val)
    clipped[clipped != max_val] = 1
    clipped[clipped == max_val] = 0
    return clipped


def compute_area(mask, pixel_dimensions):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        lung_mask: binary lung mask
        pixel_dimensions: list or tuple with two values
    Returns: the lung area in mm^2
    :param pixel_dimensions:
    :param mask:
    """
    mask[mask >= 1] = 1
    lung_pixels = np.sum(mask)
    return lung_pixels * pixel_dimensions[0] * pixel_dimensions[1]


def denoise_vessels(lung_contour, vessels):
    vessels_coords_x, vessels_coords_y = np.nonzero(vessels)  # get non zero coordinates
    for contour in lung_contour:
        x_points, y_points = contour[:, 0], contour[:, 1]
        for (coord_x, coord_y) in zip(vessels_coords_x, vessels_coords_y):
            for (x, y) in zip(x_points, y_points):
                d = euclidean_dist(x - coord_x, y - coord_y)
                if d <= 0.1:
                    vessels[coord_x, coord_y] = 0
    return vessels


def plot_3d(image, threshold=700, color="navy"):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
