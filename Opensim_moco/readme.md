	# Opensim Moco Simulation Guidelines

	**Editor:** Haedo Cho  
	**Contact:** hcho@g.harvard.edu  

	This folder includes OpenSim Moco simulations for three tasks:
	1. Sit-to-stand
	2. Incline walk
	3. Lifting an object

	---

	## Useful source

	Refer to the [OpenSim Moco Forum](https://simtk.org/plugins/phpBB/indexPhpbb.php?group_id=1815&pluginname=phpBB) for useful resources and support.

	---

	## Recommended Steps for Simulation Convergence

	### Step 1: Minimize Kinematics Tracking Error
	- Set global control effort weight to `0` (ignore muscle activation and residual actuator penalties).
	- Ignore contact dynamics.
	- If the simulation converges, tracking error should be near zero.
	- Note: High residual forces/torques (especially pelvis y-axis) may occur due to lack of contact compensation.

	### Step 2: Add Muscle Activation Penalty
	- After achieving kinematic convergence, set global control effort weight to `1` (including muscle activation).
	- Start with a muscle activation control weight (penalty) of `1`.
	- If muscle activation is unreasonably low (<10%), reduce the control weight to `0.001`.
	- Residual forces/torques may still be high, as contact dynamics are not yet included.

	### Step 3: Incorporate Contact Dynamics
	- Once muscle activation + kinematics converge, implement contact dynamics.
	- Use contact spheres (e.g., on the foot) and half spaces (e.g., floor or chair).
	- Ensure reasonable contact without excessive penetration.

	### Step 4: Minimize Residual Forces and Torques
	- When step 3 converges, minimize residuals (to bridge experimental vs. simulation discrepancies).
	- Residual values should not exceed acceptable ranges.
	- For guidelines, see:  
	  Hicks, J.L., Uchida, T.K., Seth, A., Rajagopal, A., & Delp, S.L. (2015). Is my model good enough? Best practices for verification and validation of musculoskeletal models and simulations of movement. *Journal of Biomechanical Engineering, 137*(2), p.020905.

