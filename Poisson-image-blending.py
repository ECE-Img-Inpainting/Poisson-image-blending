import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def translation(img, x, y):
    '''x and y are the position that you want to place the center of the source on'''
    source_height = source.shape[0]
    source_width = source.shape[1]
    dx = x - source_width/2
    dy = y - source_height/2
    matrix = np.float32([[1,0,dx], [0,1,dy]])
    dimensions = (target.shape[1], target.shape[0])
    return cv.warpAffine(img, matrix, dimensions)
    
def laplacian_matrix(height, width):
    # construct mat_D
    mat_D = sparse.lil_matrix((width, width))
    mat_D.setdiag(4, 0)
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(-1, 1)
    # construct mat_A
    mat_A = sparse.block_diag([mat_D] * height) # mat_A now is a coo_matrix
    mat_A = mat_A.tolil() # convert it to lil_matrix
    mat_A.setdiag(-1, width)
    mat_A.setdiag(-1, -width)
    return mat_A

def poisson_blending(source, mask, target):

    mask[mask != 0] = 1
    height = target.shape[0]
    width = target.shape[1]

    source = translation(source, 280, 210)
    mask = translation(mask, 280, 210)

    mat_A = laplacian_matrix(height, width)
    laplacian = mat_A
    print(mask.shape)
    print(source.shape)
    print(target.shape)
    # mask = mask[:height,:width]

    # do not apply laplacian operator to pixels outside the mask
    # so set corresponding row to be identity
    for y in range(1, height - 1): # skip the marginal pixels
        for x in range(1, width - 1):
            if mask[y, x] == 0:
                k = x + y * width
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + width] = 0
                mat_A[k, k - width] = 0
    mat_A = mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(3): # calculate each channel separately
        source_flat = source[:,:,channel].flatten()
        target_flat = target[:,:,channel].flatten()
        # inside the mask
        mat_b = laplacian.dot(source_flat)
        # outside the mask
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
        x = spsolve(mat_A, mat_b)  # solve the least square problem 
        x = x.reshape((height, width))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target[:,:,channel] = x
    return target

def simple_blending(source, mask, target):
    source = translation(source, 300, 200)
    mask = translation(mask, 300, 200)
    mask[mask != 0] = 1

    return (source * mask) + (target * (1 - mask))

source = cv.imread('/Users/zhaosonglin/Desktop/latex project/Poisson Image Editing/images/source_03.jpg')
target = cv.imread('/Users/zhaosonglin/Desktop/latex project/Poisson Image Editing/images/target_03.jpg')
mask = cv.imread('/Users/zhaosonglin/Desktop/latex project/Poisson Image Editing/images/mask_03.jpg', cv.IMREAD_GRAYSCALE)

# plt.subplot(1, 3, 1)
# plt.imshow(source[:,:,::-1])
# plt.title('source')
# plt.subplot(1, 3, 2)
# plt.imshow(target[:,:,::-1])
# plt.title('target')
# plt.subplot(1, 3, 3)
# plt.imshow(mask)
# plt.title('mask')
# plt.show()

# blended = simple_blending(source, mask, target)
blended = poisson_blending(source, mask, target)
cv.imwrite(('poisson2.png'), blended)
cv.imshow('blended', blended)
cv.waitKey(0)
cv.destroyAllWindows()

