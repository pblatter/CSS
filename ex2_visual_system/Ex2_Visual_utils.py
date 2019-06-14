'''
Simulation of a retina implant. 

Run file Ex_2_Visual.py

Lets the user choose an image that will be used for both tasks. 

Part 1: Retinal Ganglion Cells
The user has to select a point in the picture from which the distance to the corner that
is futhest away will be computed. This distance is divided into NB_ZONES equidistant
zones which divide the input image into 10 circular zones. For the middlepoints 
of these zones, the parameters for the Difference of Gaussians (DoG) filter are computed. 
Every zone is convoluted with the correct DoG filter.


Part 2: Cells in V1

2D-Gabor filter are computed for different angles (0 - 30 - 60 - 90 - 120 - 150 deg).
The filters are then applied to the original image, and combined to get a
final "combined image".

Some constants are assumed and defined in global_variables.py

Authors: Jan Wiegner, Diego Gallegos Salinas, Philippe Blatter
Version: 6
Date: 27.04.2019
'''

# Standard imports
import numpy as np 
from numpy import pi,sin,cos,exp
import math
import os

# Special imports
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from tkinter import filedialog, Tk
import cv2



# Local modules
from global_variables import *
from UI import UI


def build_circular_zones(max_dist, boundary_dists, image_shape, focus_point):
    '''
    Creates a mask consisting of different ciruclar zones in which to apply the Gabor filters.

    Arguments: 
        - max_dist: maximal distance from focus point to corner
        - boundary_dists: distances to outer boundaries of zones
        - image_shape: shape of input image
        - focus_point: coordinates of focus point

    Returns: 
        - Z: image of the same shape as input image containing the circular zones
    '''
    
    x_focus = focus_point[0]
    y_focus = focus_point[1]

    x_range = np.arange(image_shape[0])
    y_range = np.arange(image_shape[1])

    X,Y = np.meshgrid(y_range, x_range)

    Z = np.zeros(X.shape)
    R = np.sqrt((X-x_focus)**2 + (Y-y_focus)**2) #distances from focus point to all points

    for i, dist in enumerate(boundary_dists): #set values of pixels to zone number
        Z[R>dist] = i

    return Z

def geometry_pipeline(displacement_pix):
    '''
    Computes the parameters for the DoG filters

    Projects distance from focus onto retina, calculates rfs, projects
    rfs back onto image and calculates parameters from that.
    Assumes that the center of the eye is aligned with the fovea and
    focus point on the image.
    
    Arguments: 
        - displacement_pix: center point of the specific region in pixel

    Returns: 
        - sigma_1: sigma_1 for the DoG filter
        - sigma_2: sigma_2 for the DoG filter
        - sidelength: side length of the filter kernel in pixels
    '''

    px_per_cm = PX_PER_CM

    # displacement in cm
    displacement_cm = 1/px_per_cm * displacement_pix

    # computation of angle alpha == length of arc on unit circle
    alpha = math.atan(displacement_cm/DIST_TO_IMAGE_CM)

    # computation of eccentrity in mm == length of arc on eye
    eccentrity_mm = alpha * R_EYE_MM

    # computation of RFS [arcmin] uging provided formula
    rfs_arcmin = 10 * eccentrity_mm

    # computation from [arcmin] -> [radians]: x_arcmins * (PI / (60*180)) = y_radians
    rfs_radians = rfs_arcmin * (pi/(60*180))

    # project rfs onto image
    side_length_cm  = math.tan(rfs_radians) * DIST_TO_IMAGE_CM
    side_length_pix = side_length_cm * px_per_cm

    # computation of sigma_1 and sigma_2
    sigma_1 = side_length_pix/10
    sigma_2 = 1.6 * sigma_1

    return sigma_1, sigma_2, side_length_pix

def compute_region_distances(max_dist, verbose=False):
    '''
    Divides the maximal distance into NB_ZONES equidistant regions

    Arguments: 
        - max_dist: maximal distance from focus point to image corner
        - verbose: if detailed output should be given to console

    Returns:
        - center_dists: array containing distances to center of the regions
        - boundary_dists: array containing distances to boundaries of the regions
    '''
    zone_width = max_dist / NB_ZONES
    mid_dist = zone_width / 2

    center_dists = []
    boundary_dists = [0.0]
    for i in range(NB_ZONES):
        center_dists.append(i*zone_width + mid_dist)
        boundary_dists.append((i+1)*zone_width)

    if verbose:
        print(f'max dist: {max_dist}')
        print(f'zone_width: {zone_width}')
        print(f'mid dist: {mid_dist}')
        print(f'distances to centers of zones: {center_dists}')
        print(f'distances to boundaries of zones: {boundary_dists}')

    return center_dists, boundary_dists

def compute_max_dist(image, focus_point, verbose=False):
    '''
    Computes the maximum distance from the focus point to the 
    farthest corner of the image.

    Arguments: 
        - image: image 
        - focus_point: coordinates of focus point
        - verbose: if detailed output should be given to console

    Returns: 
        - max_dist: maximal distance from focus point to corner
        - corner_coords: coordinates of the corner that is the farthest away
    '''
    x = focus_point[0]
    y = focus_point[1]

    # corners of image
    zero = 0
    height = image.shape[0]    
    width = image.shape[1]


    
    d1 = math.sqrt((x-zero)**2 + (y-zero)**2) # distance to top left corner
    d2 = math.sqrt((width-x)**2 + (y-zero)**2) # distance to top right corner
    d3 = math.sqrt((width-x)**2 + (height-y)**2) # distance to bottom right
    d4 = math.sqrt((x-zero)**2 + (height-y)**2) # distance to bottom left

    # coordinates only stored for visualization -> -1 so that the image stays in place
    distances = {d1:('top left corner',0,0), d2:('top right corner',width-1,0), d3:('bottom right corner',width-1,height-1), d4:('bottom left corner',0,height-1)}

    max_dist = max(d1,d2,d3,d4)
    corner_coords = [distances[max_dist][1], distances[max_dist][2]]  

    # Output of values to console
    if verbose:
        print(f'zero point: {zero}')
        print(f'height: {height}')
        print(f'width: {width}')
        print(f'focus_point: {focus_point}')
        print(f'farthest corner: {distances[max_dist]}')      
       
    return max_dist, corner_coords

def plot_midpoints(corner_coords, focus_point, dists):
    '''
    Plots the midpoints of the circular zones

    Arguments: 
        - corner_coords : coordinates of corner
        - focus_point : coordinates of focus point
        - dists : vector of ditances from focus point
    '''
    x_corner = corner_coords[0]
    y_corner = corner_coords[1]

    x_focus = focus_point[0]
    y_focus = focus_point[1]

    # normalized vector from focus to corner
    # end point = corner, start point = focus
    vect = np.array([x_corner - x_focus, y_corner - y_focus])
    vect = vect/np.linalg.norm(vect)

    dists = np.array(dists)

    xx = x_focus + vect[0]*dists
    yy = y_focus + vect[1]*dists

    plt.scatter(xx,yy,marker='.', color='blue', zorder=3)

def vizualize_zones(img, zones, focus_point, corner_coords, center_dists):
    '''
    Vizualize zones and points 

    Arguments: 
        - zones: image of the same shape as the input image containing the circular zones
        - corner_coords : coordinates of corner
        - focus_point : coordinates of focus point
        - dists : vector of ditances from focus point

    '''
    plt.figure(2)
    #show image
    plt.imshow(img, cmap=plt.cm.gray, interpolation='bilinear')
    #plot zones
    plt.imshow(zones, cmap=plt.cm.gray, alpha=0.7, interpolation='bilinear')
    #plot diag
    plt.plot((focus_point[0],corner_coords[0]), (focus_point[1],corner_coords[1]), 'ro-')
    # plot midpoints of zones for visualization
    plot_midpoints(corner_coords, focus_point, center_dists)
    plt.title('Visualization of zones, and midpoints')
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    plt.draw()

def create_filter_list_DoG(center_dists, verbose=False):
    '''
    Creates a list of filters depending on the different parameters computed for every zone.

    Arguments: 
        - center_dists: list of center points of the circular zones
        - verbose: if detailed output should be given to console

    Returns: 
        - filters: list of filter functions
    '''

    # Function that creates a Difference of Gaussians filter
    # Convert to floats to not have problems with uint8 substraction underflow
    DoG = lambda k_s, s1, s2: (lambda img :  cv2.GaussianBlur(img,ksize=(k_s,k_s),sigmaX=s2,sigmaY=s2).astype(float) 
                                           - cv2.GaussianBlur(img,ksize=(k_s,k_s),sigmaX=s1,sigmaY=s1).astype(float))
        

    filters = []
    for dist in center_dists:
        sigma_1, sigma_2, side_length = geometry_pipeline(dist)
        
        #Get nearest odd number for kernel size, and at least 3
        kernel_size = (2*np.floor(side_length/2)+1).astype(int)
        kernel_size = max(3,kernel_size)

        filters.append(DoG(kernel_size, sigma_1, sigma_2))

        if verbose:
            print(f'sigma_1 : {sigma_1}')
            print(f'sigma_2: {sigma_2}')
            #print(f'side_length_pix: {side_length}')
            print(f'kernel_size: {kernel_size}\n')   


    return filters

def create_kernel_list_gabor(thetas):
    '''
    Creates a list of filters depending on the different parameters computed for every zone.

    Based on code from gabor_demo.py

    Arguments: 
        - thetas: list of angles in radians for the gabor filters

    Returns: 
        - kernels: list of kernels
    '''

    #These parameters were chosen based on testing with gabor_demo.py

    #ksize = 21
    #sigma = 0.2
    #lambd = 0.5
    #gamma = 0.5
    #psi = pi/2

    ksize = 13
    sigma = 0.2
    lambd = 2.0
    gamma = 0.5
    psi = pi/2
    
    xs=np.linspace(-1., 1., ksize)
    ys=np.linspace(-1., 1., ksize)
    x,y = np.meshgrid(xs,ys)

    kernels = []
    for theta in thetas:
        x_theta =  x*cos(theta) + y*sin(theta)
        y_theta = -x*sin(theta) + y*cos(theta)

        kernel = np.array(exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*cos(2.*pi*x_theta/lambd + psi),dtype=np.float32)

        kernels.append(kernel)

    return kernels

def apply_filters(img, Filters, Zones, verbose=False):
    '''
    First applies every filter to the input image and stores the filtered images in a list. 
    Then builds a final output image where the correct filter is applied to the correct zone.

    Arguments: 
        - img: input image
        - Filters: list of filters
        - Zones: image of the same shape as the input image containing the circular zones
        - verbose: if detailed output should be given to console

    Returns: 
        - final: final output i.e. correct filter applied to correct zone
    '''

    Filtered = []

    # apply every filter to the image once
    
    for Filter in Filters:
        f_img = Filter(img)
        Filtered.append(f_img)

        if verbose:
            print(f'max of filtered image: {f_img.max()}')
            print(f'min of filtered image: {f_img.min()}\n')

    if verbose:
        print(f'max of zone: {Zones.max()}')
        print(f'min of zone: {Zones.min()}\n')

    final = np.zeros(img.shape) #, dtype = np.uint8)
    for ii in np.arange(NB_ZONES):
        final[Zones==ii] = Filtered[ii][Zones==ii]

    return final

def main(in_path=None, verbose=False):
    '''
    Main function that coordinates the entire process.

    Handles file input/output and calls tasks 1 and 2
    '''

    #Let user choose file graphically if they did not run from file EX_2_Visual.py
    if(in_path == None):
        print("Warning: Run file EX_2_Visual.py to run tasks with UI")
        root = Tk()
        in_path =  filedialog.askopenfilename(initialdir = "./Images/",title = "Select file",
                                                filetypes = (("image files","*.jpg *.png *.tif *.bmp"),("all files","*.*")))
        root.destroy()

    #Extract filename
    (head, tail) = os.path.split(in_path)
    (fname, ext) = os.path.splitext(tail)

    # Read in file
    img = plt.imread(in_path)
    print (f"Opened file: {in_path}")
    #Transform to grayscale
    img = rgb2gray(img)
    print(f'shape of input image: {img.shape}')

    #Run task 1
    #verbose = False
    img_on, img_off = task1(img, verbose)

    # Write outputs of task 1 to disk
    cv2.imwrite(f'./out/{fname}_task1_ON.png',img_on)
    cv2.imwrite(f'./out/{fname}_task1_OFF.png',img_off)
    print ('Wrote file to:' + f'./out/{fname}_task1_ON.png')
    print ('Wrote file to:' + f'./out/{fname}_task1_OFF.png')

    img_gab = task2(img, verbose)
    cv2.imwrite(f'./out/{fname}_task2_gabor.png',img_gab)
    print ('Wrote file to:' + f'./out/{fname}_task2_gabor.png')

def task1(img=[], verbose=False):
    '''
    Function that runs code for task 1: Simulation of Retinal Ganglion Cells

    The user has to select a point in the picture from which the distance to the corner that
    is futhest away will be computed. This distance is divided into NB_ZONES equidistant
    zones which divide the input image into 10 circular zones. For the middlepoints 
    of these zones, the parameters for the Difference of Gaussians (DoG) filter are computed. 
    Every zone is convoluted with the correct DoG filter.

    Arguments: 
        - verbose: if detailed output should be given to console

    Returns: 
        - ganlion_on: image showing the activation of ON ganglion cells
        - ganlion_off: image showing the activation of OFF ganglion cells
    '''
    # show the image
    plt.imshow(img, cmap=plt.cm.gray, interpolation='bilinear')
    plt.title('Select focus point')
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    # define number of pixels that should be selected and get the respective coordinates
    n_pixel = 1
    selected = np.round(np.array(plt.ginput(n_pixel))) # returns numpy ndarray [n_pixel, 2]
    plt.close()
    focus = selected[0]

    # compute the distance to the farthest corner
    max_dist, corner_coords = compute_max_dist(img, focus, verbose)

    # divide the maximal distance into NB_ZONES equidistant zones 
    # and compute the center points of the regions
    center_dists, boundary_dists = compute_region_distances(max_dist, verbose)


    # build the circular zones
    zones = build_circular_zones(max_dist, boundary_dists, img.shape, focus)
    

    # create list of DoG filters
    filters = create_filter_list_DoG(center_dists, verbose)

    # apply the filters and compose according to zones
    filtered = apply_filters(img, filters, zones, verbose)


    #Simulate the cells by making them fire at max if there is a signal

    # Thresholds for gaglions firing
    # Higher tresholds lead to no response near focus point
    #threshold_on = np.max(filtered)/20 
    #threshold_off = np.min(filtered)/20
    threshold_on = 0
    threshold_off = 0


    # Positive values are responses of ON cells
    ganlion_on = np.zeros_like(filtered)
    ganlion_on[filtered<0] = 0
    ganlion_on[filtered>threshold_on] = 255

    # Negative values are responses of ON cells
    ganlion_off = -np.zeros_like(filtered)
    ganlion_off[filtered<threshold_off] = 255
    ganlion_off[filtered>0] = 0

    if verbose:
        print(f'max of composed filtered image: {filtered.max()}')
        print(f'min of composed filtered image: {filtered.min()}')
        print(f'max of ganlion_on: {ganlion_on.max()}')
        print(f'min of ganlion_on: {ganlion_on.min()}')
        print(f'max of ganlion_off: {ganlion_off.max()}')
        print(f'min of ganlion_off: {ganlion_off.min()}')


    # Plot cell responses
    fig = plt.figure(1, figsize=(12,5))
    fig.suptitle('Responses of cells on retina')

    plt.gray()
    plt.subplot(1, 2, 1)
    plt.imshow(ganlion_on, interpolation='bilinear')
    plt.title('ON Ganglion cell responses')
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    plt.subplot(1, 2, 2 )
    plt.imshow(ganlion_off, interpolation='bilinear')
    plt.title('OFF Ganglion cell responses')
    plt.xlabel('width (px)')

    plt.subplots_adjust(left=0.06, right=0.98, wspace=0.1)
    plt.draw()

    print("Showing cell responses. Click or press key on plot to continue.")
    
    # Avoid exceptions from user closing window
    try: 
        plt.waitforbuttonpress(0)
    except:
        pass

    # Vizualize zones
    vizualize_zones(img, zones, focus, corner_coords, center_dists)

    print("Showing zone visualization. Click or press key on plot to continue.")
    # Avoid exceptions from user closing window
    try: 
        plt.waitforbuttonpress(0)
    except:
        pass
    plt.close(1)
    plt.close(2)

    return ganlion_on, ganlion_off

def task2(img=[], verbose=False):
    '''
    Function that runs code for task 2: Simulation of Cells in V1

    2D-Gabor filter are computed for different angles.
    The filters are then applied to the original image, and combined to get a
    final "combined image".

    Arguments: 
        - verbose: if detailed output should be given to console

    Returns: 
        - averaged: average of images filterd with gabor functions
    '''
    
    # Define angles at wich to create gabor kernels
    thetas = np.array([0, 30, 60, 90, 120, 150]) * pi/180

    # Create gabor kernels
    kernels = create_kernel_list_gabor(thetas)

    # Apply gabor kernels
    apply_kernel = lambda k: cv2.filter2D(img, -1, k)
    filtered = np.array(list(map(apply_kernel, kernels)))

    # Take average over all filtered images, and discard values less than 0
    averaged = np.mean(filtered, axis=0)
    averaged[averaged<0] = 0.0
    
    # Scale to be in integers
    averaged = np.uint8(averaged/np.max(averaged) * 255)

    plt.figure(3, figsize=(8,6))

    plt.gray()
    plt.imshow(averaged, interpolation='bilinear')
    plt.title('Average of images filterd with gabor functions')
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    plt.draw()
    print("Showing average of images filterd with gabor functions. Click or press key on plot to continue.")
    try: 
        plt.waitforbuttonpress(0)
    except:
        pass
    plt.close(3)
    
    return averaged

if __name__ == '__main__':
    main()