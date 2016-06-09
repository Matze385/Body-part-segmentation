import skimage as skim
from sampling import *
from skimage.transform import rotate 



if __name__ == "__main__":
    b = np.zeros((3,3))
    b[1,2]= 1
    b[0,2]= 2
    b[2,1]=1
    b[1,2]=2
    print b
    c = selectPixelsBalanced(b, 1, 2)
    print c
    print (rotate(c,45, resize=False, center=(1,1), order=0, preserve_range=True))
    
    """
    a = np.zeros((10,10), dtype = np.uint8)
    a[2,2] =1
    a[2,3] =1
    a[5,5] =1
    a[5,6] =1
    print a
    selected_map_rot = rotate(a, 10, resize = False, center = (5,5),order = 0,preserve_range=True)
    print(selected_map_rot.astype(np.uint8))
"""
