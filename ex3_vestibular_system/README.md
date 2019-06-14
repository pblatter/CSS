## Task 1

1. Definition of approximate rotation matrix of 90° along x-axis

    R_approx = [1  0  0
                0  0 -1
                0  1  0 ]

2. Computation of quaternion describing the rotation of g=(0/9.81/0) to g'=(a/b/c) 

https://moodle-app2.let.ethz.ch/pluginfile.php/677556/mod_resource/content/0/IMU-rotations_annotated.pdf

    alpha = acos((g*g')/(|g|*|g'|))
    m = (g x g)/|g x g'|

    -> q = m*sin(alpha/2)

3. Convert either R_approx into the format of q or the other way around "combine" R_approx and q in order
   to get the entire rotation that maps points from the sensor space into world coordinates. 
   
   Take the inverse to obtain R_hc_IMU: the transformation from "head coordinates" / "world coordinates at t=0" to IMU coordinates.
   R_hc_IMU = ( R(q) * R_approx )^T   (as rotation matricies)
   (Rotation matricies are orthogonal, so transpose == inverse)
   
   Define R_rc_IMU: the transformation from "Reid's Line coordinates" to IMU coordinates.
   R_rc_IMU = R_hc_IMU * R_y15  (as rotation matricies)
   (Calculated by mutyplying R_rc_IMU from right by a rotation matrix doing a rotation of +15 degrees along the y-axis)


4. Vector orthogonal to SCC_v=(0.32269/-0.03837/-0.94573) in a space, where the horizontal (xy) plane is defined
   by Reid's Line. 

   -> R_rc_IMU*SCC_v = "vector orthogonal to SCC in IMU coordinates" = SCC_v_IMU
   
   
   Ot_dir_IMU = R_hc_IMU*[0,1,0]

5. Stimulations: 
	SCC stimulations:
		SCC_stim(t) = dot(omega(t), SCC_v_IMU) (dot product)
		(We obtain omega(t) from the scikit-kinematics library)
		
	Otolith stimulation:
		Ot_stim(t) = dot(acc(t), Ot_dir_IMU) (dot product)
		(We obtain acc(t) from the scikit-kinematics library)
	
	-> maybe a plot stimulations = f(time) ?

6. Cupula Displacement for Head Movements -> slide (42/72) from the lecture notes 
	Math:
   https://en.wikibooks.org/wiki/Sensory_Systems/Computer_Models/Vestibular_Simulation
   Code:
   https://nbviewer.jupyter.org/github/thomas-haslwanter/CSS_ipynb/blob/master/Vestibular_2_SCC_Transduction.ipynb
   
    In short:
    Define a transfer function.
    Simulate output of the continuous-time linear system.
    From this we get displacement in radians, transform them to mm

   Write the maximal cupular displacements (positive and negative) to CupularDisplacement.txt

7. Calculate the minimum and maximum acceleration along the Ot_dir_IMU direction in m/s^2, and write to MaxAcceleration.txt.

	Project acceleration vector onto Ot_dir_IMU vector and determine vector norm 
	https://en.wikipedia.org/wiki/Vector_projection
	(check that Ot_dir_IMU has norm 1)

	dir_acc(t) = dot(acc(t),Ot_dir_IMU) * Ot_dir_IMU
	max_t/min_t  norm ( dir_acc(t) )
	
	Projectiong on a unit vector and taking the norm of that is equivalent to
    simply taking taking the dot product between the two.
    
    max_t/min_t  norm ( dir_acc(t) ) = max_t/min_t  dot(acc(t),Ot_dir_IMU)
                                     = max_t/min_t  Ot_stim(t)
    


## Task 2

1. Nose orientation at t=0 -> nose = [1,0,0] in space coordinates

	R_IMU_hc = (R_hc_IMU)^T
	
	rotate angular velocity measurements with R_IMU_hc
	
2. Update steps w.r.t. angular velocity measurements. (vis slide 60/72)

	q(t) = q(t0) * delta~q(t1) * ... * delta~q(tn)

	Output graph of quaternion components
	
	Visualize with this code:
	https://nbviewer.jupyter.org/github/thomas-haslwanter/CSS_ipynb/blob/master/Vestibular_3D_Animation.ipynb
	
	