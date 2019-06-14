"""

Exercise: Simulation of a Vestibular Implant


Authors: Jan Wiegner, Philippe Blatter, Diego Gallegos Salinas
Version: 6
Date: 09.06.2019


Naming convention in this file:
    
Rotation matricies start with R
quaternions start with q

R_a_b is rotation from coordinates a to coordinates b
name_a is a vector in coordinates a

approx: approximate IMU coordinates
IMU: IMU coordinates (all measurements are in these coordinates)
hc: head coordinates / 'world coordinates at t=0'
rc: Reid's line coords

"""

from skinematics.sensors.xsens import XSens
import numpy as np
from numpy import arccos, cross, sin
from numpy.linalg import norm
import skinematics as sk
import os

import matplotlib.pyplot as plt

from scipy import signal



'''
############################################
 _____             _           _ 
(_   _)           ( )        /' )
  | |   _ _   ___ | |/')    (_, |
  | | /'_` )/',__)| , <       | |
  | |( (_| |\__, \| |\`\      | |
  (_)`\__,_)(____/(_) (_)     (_)
############################################
    '''
def task1():
    """Simulate the neural vestibular responses during walking:
    
    Calculate the maximum cupular displacements (positive and negative)
    and write them to CupularDisplacement.txt
    
    Calculate the minimum and maximum acceleration along a given otolith direction
    and write them to MaxAcceleration.txt
    """
    
    #Read in the data
    in_file = r'./MovementData/Walking_02.txt'
    out_dir = r'./out'
    try:
        os.mkdir(out_dir)
        print("Created directory " , out_dir) 
    except FileExistsError:
        pass

    sensor = XSens(in_file=in_file)
    
    #Extract data from sensor
    N = sensor.totalSamples
    sample_rate = sensor.rate
    # (N * 3) dimensional array of accelerations 
    accs = sensor.acc
    # (N * 3) dimensional array of omegas
    omegas = sensor.omega


    ### 1.
    
    # Transform from head coordinates / world coords to Approximate IMU coords
    #Equivalent to:
    R_hc_approx = sk.rotmat.R(axis='x', angle=90)
    
    g_hc = np.array([0, 0, -9.81])
    g_approx = R_hc_approx @ g_hc # [0.  , 9.81, 0.  ]


    ### 2. 

    # Assume acceleration vector at time 0 is only gravity
    g_IMU = accs[0] # [ 4.37424   8.578849 -1.814515]

    

    # Define quaternion that defines the smallest rotation between g_approx and g_IMU
    alpha = arccos((np.dot(g_approx,g_IMU))/(norm(g_approx)*norm(g_IMU)))
    m = cross(g_approx,g_IMU)/norm(cross(g_approx,g_IMU))

    q_approx_IMU = sk.quat.Quaternion(m*sin(alpha/2)) # quaternion approx -> IMU
    R_approx_IMU = q_approx_IMU.export('rotmat') #-> in case one wants to do the computations using matrices
    

    ### 3. 

    # Transformation from 'head coordinates' / 'world coordinates at t=0' to IMU coords 
    # Rotation matricies should be interpreted from right to left
    # @ is matrix multiplication in numpy, * is elementwise
    R_hc_IMU = R_approx_IMU @ R_hc_approx  # transform hc -> approx -> IMU

    
    R_rc_hc = sk.rotmat.R(axis='y', angle=15) # Reid's line coords (rc) -> head coords (hc)
    R_rc_IMU = R_hc_IMU @ R_rc_hc # rc -> hc -> IMU

                    
    # Semi circular canal vector in Reid's line coordinates
    SCC_v= np.transpose(np.array([0.32269,-0.03837,-0.94573]))
    
    # Otolith direction vectors
    Otolith_dir_hc = np.transpose(np.array([0,1,0]))

    ### 4. 

    # Transform vectors to IMU coordinates
    SCC_v_IMU = R_rc_IMU @ SCC_v 
    Otolith_dir_IMU = R_hc_IMU @ Otolith_dir_hc
    


    ### 5. 

    # SCC stimulation
    # [Nx3] * [3x1] \in [Nx1] -> one value for every time step
    SCC_stim_all = []
    for i in range(N):
        SCC_stim = np.dot(np.transpose(omegas[i]), SCC_v_IMU) 
        SCC_stim_all.append(SCC_stim)


    # Otolith stimulation
    # [Nx3] * [3x1] \in [Nx1] -> one value for every time step
    Ot_stim_all = []
    for i in range(N):
        Ot_stim = np.dot(np.transpose(accs[i]), Otolith_dir_IMU) 
        Ot_stim_all.append(Ot_stim)
        
    ### 6. 

    # Cupula displacement for head movements

    # SCC dynamics
    T1 = 0.01
    T2 = 5
    
    # Define transfer function
    num = [T1*T2, 0]
    den = [T1*T2, T1+T2, 1]
    scc_transfer_function = signal.lti(num,den)
    
    # Radius of SCC
    radius = 3.2 #mm
    
    # Create time axis with length and increments of sensor data
    t = np.arange(0, 1./sample_rate*N, 1./sample_rate)

    # Estimate displacement (radians) with calculated SCC_stim_all
    _, out_sig, _ = signal.lsim(scc_transfer_function, SCC_stim_all, t)
    
    # radians -> mm
    cuppula_displacements = out_sig * radius
    
    
    #For visualization
#    plt.hist(cuppula_displacements, bins=100)
#    plt.show()


    max_pos = np.max(cuppula_displacements)
    max_neg = np.min(cuppula_displacements)
    print('Maximal positive cupular displacement:', max_pos, 'mm')
    print('Maximal negative cupular displacement:', max_neg, 'mm')
    with open(f'{out_dir}/CupularDisplacement.txt', 'w+') as f:
        f.write(f'Maximal positive cupular displacement: {max_pos}\n')
        f.write(f'Maximal negative cupular displacmenet: {max_neg}\n')

    print('Wrote values to CupularDisplacement.txt')
    

    ### 7.

    # Minimum / maximum acceleration along Ot_dir_IMU direction [m/s²]
    # Projection of acceleration vector onto Ot_dir_IMU, then determine vector norm
	# https://en.wikipedia.org/wiki/Vector_projection
    # dir_acc(t) = dot(acc(t),Ot_dir_IMU) * Ot_dir_IMU
    # max_t/min_t  norm ( dir_acc(t) )
    
    # Projectiong on a unit vector and taking the norm of that is equivalent to
    # simply taking taking the dot product between the two.
    # (same calculation as Ot_stim_all)
    

    norms = Ot_stim_all
    
    max_acc = np.max(norms)
    min_acc = np.min(norms)
    print('Maximal acceleration along otolith direction:',max_acc, 'm/s²')
    print('Minimal acceleration along otolith direction:',min_acc, 'm/s²')
    
    with open(f'{out_dir}/MaxAcceleration.txt', 'w+') as f:
        f.write(f'Maximal acceleration along otolith direction: {max_acc} m/s²\n')
        f.write(f'Minimal acceleration along otolith direction: {min_acc} m/s²\n')
    
    print('Wrote values to MaxAcceleration.txt')

    return R_hc_IMU

'''
###########################################
 _____             _           __   
(_   _)           ( )        /'__`\ 
  | |   _ _   ___ | |/')    (_)  ) )
  | | /'_` )/',__)| , <        /' / 
  | |( (_| |\__, \| |\`\     /' /( )
  (_)`\__,_)(____/(_) (_)   (_____/'
###########################################                                    
                                    
'''
   
def task2(R_hc_IMU):
    """ Calculate the orientation of the "Nose"-vector 
    
    Plot quaternion values
    
    Plot quaternion vector values, save orientations to video
    and output the orientation at the end of walking the loop
    """

    out_video_file = './out/task2_out.mp4'
    out_plot_file = "./out/task2_out.png"

    R_IMU_hc = np.transpose(R_hc_IMU)
    
    Nose_init_hc = np.transpose(np.array([1,0,0]))
    
    #Read in sensor data
    in_file = r'./MovementData/Walking_02.txt'
    sensor = XSens(in_file=in_file)
        
    N = sensor.totalSamples
    sample_rate = sensor.rate
    # (N * 3) dimensional array of omegas
    omegas = sensor.omega

    # Transform omegas to head coordinates
    omegas_hc = []
    for i in range(N):
        omega_hc = R_IMU_hc @ np.transpose(omegas[i])
        omegas_hc.append(np.transpose(omega_hc))
        
    # (N * 3) dimensional array of omegas in head coordinates
    omegas_hc = np.array(omegas_hc)

    
    # Calculate all orientation quaternions
    qs = -sk.quat.calc_quat(omegas_hc, [0,0,0], sample_rate, 'bf')
    
    
    # Output of last orientation of nose
    q_last = qs[-1,:]
    R_last = sk.quat.convert(q_last)
    
    Nose_end_hc = R_last @ Nose_init_hc
    print('Nose orientation at the end of the walking the loop:', Nose_end_hc)


    # Graph of all quaternion components
    # Only plot vector part
    plt.plot(range(N), qs[:,1])
    plt.plot(range(N), qs[:,2])
    plt.plot(range(N), qs[:,3])
    plt.savefig(out_plot_file)
    print('Plot image saved to', out_plot_file)
    plt.show()


    # Create moving plot of nose vector
    # Use scikit-kinematics visualizations  
    # (Need ffmpeg)
    print('Creating animation of orientations...')
    sk.view.orientation(qs, out_video_file, 'Nose orientation', deltaT=1000./sample_rate)


def main():
    """
    """
    R_hc_IMU = task1()
    task2(R_hc_IMU)

if __name__ == '__main__':
    main()
    
    
"""
Console output:

Maximal positive cupular displacement: 0.15206860341896528 mm
Maximal negative cupular displacement: -0.10549995653261182 mm
Wrote values to CupularDisplacement.txt
Maximal acceleration along otolith direction: 5.6298951071964645 m/s²
Minimal acceleration along otolith direction: -6.870260679053982 m/s²
Wrote values to MaxAcceleration.txt
Nose orientation at the end of the walking the loop: [ 0.9969088  -0.05902648 -0.05185289]
Plot image saved to ./out/task2_out.png
Creating animation of orientations...
Animation saved to ./out/task2_out.mp4

"""    
    
    
    
    
