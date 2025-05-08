# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:22:27 2025

@author: Toby
"""

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('clear','')
    get_ipython().run_line_magic('reset', '-sf')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sys

######### Functions ########

class generate_points:
    
    def arc(N, R, arc_lenght_deg=360, arc_center_deg=180):
        # arc_center_deg is 0 at +x direction 90 at +y ...
        phi = np.linspace((np.pi/180)*(arc_center_deg-arc_lenght_deg/2), (arc_center_deg+arc_lenght_deg/2)*(np.pi/180), N)
        return np.array([R * np.cos(phi), R * np.sin(phi)]).T
    
    def square(N, width):
        N_per_side = int(np.round(N/4))
        # Define the four sides of the square
        x = np.linspace(-width, width, N_per_side)
        y = np.linspace(-width, width, N_per_side)
    
        # Four edges
        bottom = np.column_stack((x, -width * np.ones_like(x)))
        right  = np.column_stack((width * np.ones_like(y), y))
        top    = np.column_stack((x[::-1], width * np.ones_like(x)))  # reverse x for continuity
        left   = np.column_stack((-width * np.ones_like(y), y[::-1])) # reverse y for continuity
    
        return np.vstack((bottom, right, top, left))

def incident_field(x_m, A, k0, inc_field_dir=np.array([1, 0]), phase=0):
    nom_dir = inc_field_dir/np.linalg.norm(inc_field_dir)
    field_dist = x_m @ nom_dir

    return A * np.exp(-1j * (k0 * field_dist + phase))


def hankel_matrix(k0, x, x_l, farfield=False):
    sysM = np.zeros((len(x), len(x_l)), dtype=complex)
    for i in range(len(x_l)):
        for j in range(len(x)):
            r = np.linalg.norm(x[j] - x_l[i])
            if farfield:
                sysM[j, i] = (np.exp(-1j * k0 * r)) * np.sqrt(2 / np.pi* k0 * r)
            else:
                sysM[j, i] = sc.special.hankel2(0, k0 * r)
    return sysM

def plot_total_field(minX, maxX, minY, maxY, r_sources, S, k_0, inc_E_0,
                     inc_prop_dir=np.array([1, 0]), incident_phase=0, boundary = [], titel_ = "Log of Absolute Value of Total Field |E_tot|"):
    x = np.linspace(minX, maxX, 151, endpoint=True)
    y = np.linspace(minY, maxY, 151, endpoint=True)
    # create the mesh based on these arrays
    meshgrid = np.meshgrid(x, y)
    r_obs = np.array([meshgrid[0].flatten(), meshgrid[1].flatten()]).T
    E_inc = incident_field(x_m=r_obs, A=inc_E_0, k0=k_0, inc_field_dir=inc_prop_dir)
    E_inc_mesh = E_inc.reshape(meshgrid[0].shape)

    B = hankel_matrix(k_0, r_obs, r_sources)  # matrix between source strengths and scattered field 
    E_sca = B @ S
    E_sca_mesh = E_sca.reshape(meshgrid[0].shape)
    
    E_tot_mesh = E_inc_mesh + E_sca_mesh
    
    plt.figure()
    plt.pcolor(x, y, np.log10(np.abs(E_tot_mesh))*20,cmap='viridis')
    plt.axis("equal")
    
    plt.colorbar().set_label('E [dB]')
    if len(boundary) > 1:
        plt.scatter(boundary[:, 0], boundary[:, 1], color='w', label='boundary', s=0.5)
    # plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")


    if len(boundary) > 1:
        plt.scatter(boundary[:, 0], boundary[:, 1], color='w', label='boundary', s=0.5)

    # plt.legend()
    plt.title(f"{titel_}")
    plt.show()



def forward_scatter(N = 50, M = 100, _lambda=0.03, A = 1, sca_radius = 2.5 * 0.03, inc_field_dir = np.array([1,0]), num_measure = 120):
    
     k0 = (2 * np.pi) / _lambda 

     matching_points = generate_points.arc(M, 0.05) # Boundary points
     # matching_points = generate_points.square(M, 0.05) # Boundary points
     aux_sources = generate_points.arc(N, 0.03) # Auxilary sources 
     
     measure_points = generate_points.arc(num_measure, 0.5, 360, 0) # Measurement points
 
     E_inc = incident_field(matching_points, A, k0, inc_field_dir=inc_field_dir)  # incident field at boundary points
     
     hankel_matrix_matching = hankel_matrix(k0, matching_points, aux_sources)  # x = B
     
     ##### Solve for x in Ax = B #########
     
     normal_matrix_matching = hankel_matrix_matching.T @ hankel_matrix_matching
     
     
     C = np.linalg.solve(normal_matrix_matching, hankel_matrix_matching.T @ (-1 * E_inc))  # solves -E_inc = B @ C (B dot C) for C (with least squares without regularization)
    
    
     ########## Validate the solver #########

     # Calculating scattered at boundary to validate
     E_sca_val = hankel_matrix_matching @ C
    
     print(f"Condition number of the system forward matching matrix is: {np.linalg.cond(hankel_matrix_matching)}")
    
     plt.plot(np.abs(E_inc + E_sca_val))
     plt.title('Absolute value of total field, |E_tot|, at boundary') 
     plt.show()
     
     
     hankel_matrix_measure = hankel_matrix(k0, measure_points, aux_sources)  # system matrix between sorce strengths and scatered field values at gamma
     E_sca = hankel_matrix_measure @ C
     
     print(f"Condition number of the system forward matrix is: {np.linalg.cond(hankel_matrix_measure)}")
     
     plt.plot(np.abs(E_sca))
     plt.title('Absolute value of the scattered field, |E_sca|, at Measure Points') 
     plt.show()
     
     ######### Plots ########
     
     plt.scatter(matching_points[:, 0], matching_points[:, 1], color='r', label='Matching', s=0.5)
     plt.scatter(measure_points[:, 0], measure_points[:, 1], color='g', label='Measurement', s=0.5)
     plt.scatter(aux_sources[:, 0], aux_sources[:, 1], color='b', label='Source', s=0.5)
     plt.axis([-0.6,0.6,-0.6,0.6])
     plt.axis('equal')
     plt.xlabel("x [m]")
     plt.ylabel("y [m]")
     plt.title('Visualization of sources and points for forward') 
     # plt.legend()
     plt.show()
     
     plot_total_field(-0.15, 0.15, -0.15, 0.15, aux_sources, C, k0, A, inc_prop_dir=inc_field_dir)
     
     return E_sca, matching_points, aux_sources, measure_points, k0

if __name__ == '__main__':
    
    
    #### Constants ####
    N = 100                  
    _lambda = 0.03       # 3 cm
    A = 1
    
    field_dir = np.array([1,0]) 
    # field_dir = np.array([1,-1])
    
    E_sca_f, matching_points_f, aux_sources_f, measure_points_f, k0 = forward_scatter(N = N, 
                                                                                      M = 2 * N, 
                                                                                      _lambda = _lambda, 
                                                                                      sca_radius = 2.5 * _lambda,
                                                                                      A = A,
                                                                                      inc_field_dir = field_dir,
                                                                                      num_measure= 500
                                                                                      )
    
    
    
    ########## Inverse Solver ##############
    
    # set the position of the sources we want to use (should be different than in the forward problem)
    aux_sources_i = generate_points.arc(200, 0.01)
    
    plt.scatter(measure_points_f[:, 0], measure_points_f[:, 1], color='g', label='Measurement', s=0.5)
    plt.scatter(aux_sources_i[:, 0], aux_sources_i[:, 1], color='b', label='Source', s=0.5)
    plt.axis([-0.6,0.6,-0.6,0.6])
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title('Visualization of sources and points for inverse') 
    plt.show()
    
    # Set up the inverse problem
    hankel_matrix_i_measure = hankel_matrix(k0, measure_points_f, aux_sources_i)  # system matrix between sorce strengths and scatered field values at gamma
    print(f"Condition number of the inverse system matrix is: {np.linalg.cond(hankel_matrix_i_measure)}")
    normal_matrix_i = hankel_matrix_i_measure.T @ hankel_matrix_i_measure
    S = np.linalg.solve(normal_matrix_i, hankel_matrix_i_measure.T @ E_sca_f)  # solves Esg = G @ S  for S (with least squares )


    plot_total_field(-0.15, 0.15, -0.15, 0.15, aux_sources_i, S, k0, A, inc_prop_dir=field_dir)




















