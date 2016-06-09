import numpy as np
import random as rd

"""
select n_pix_per_class pixel per class by random and mark the positions of these pixel with number of clss in a matrix of same size as img, all other entries of this return matrix are zero
return: matrix of same size as img with ones at selected positions and zero otherwise
img: labeled image
n_pix_per_class: number of 
"""
def selectPixelsBalanced(img, n_pix_per_class, n_classes):
    assert(len(img.shape) == 2)
    x_dim = img.shape[0]
    y_dim = img.shape[1]
    img = img.reshape(x_dim*y_dim)
    idx = np.arange(x_dim*y_dim)  
    rd.shuffle(idx)                                     #resort index
    selected = np.zeros(x_dim*y_dim, dtype= np.uint8)   #selected pixels are marked with 1
    for cl in np.arange(n_classes):                     #for each class reduce shuffled idx to those that belong to labels of these class
        cl += 1
        idx_cl = filter(lambda i: cl == img[i], idx)
        assert len(idx_cl) >= n_pix_per_class
        idx_cl = idx_cl[:n_pix_per_class] 
        selected[idx_cl] = cl
    return selected.reshape((x_dim, y_dim))

   
