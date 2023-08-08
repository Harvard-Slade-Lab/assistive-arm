import os
import opensim as osim


model = osim.Model()
model.setName("inverted_pendulum")
model.set_gravity(osim.Vec3(0, -9.81, 0))

# Create rectangular cart
cart = osim.Body("body", 2.0, osim.Vec3(0), osim.Inertia(0))
cart.attachGeometry(osim.Brick(osim.Vec3(0.1, 0.05, 0.05)))


# Bar joining cart to sphere
bar = osim.Body("bar", 0, osim.Vec3(0), osim.Inertia(0))
bar.attachGeometry(osim.Cylinder(0.01, 0.3))

sphere = osim.Body("pendulum", 1.0, osim.Vec3(0), osim.Inertia(0))
sphere.attachGeometry(osim.Sphere(0.05))

pin_joint = osim.PinJoint(
    "pin",
    cart,
    osim.Vec3(0, 0, 0),
    osim.Vec3(0),
    bar,
    osim.Vec3(0, -0.3, 0),
    osim.Vec3(0),
)

weld_joint = osim.WeldJoint(
    "weld", bar, osim.Vec3(0, 0.3, 0), osim.Vec3(0), sphere, osim.Vec3(0), osim.Vec3(0)
)

cart_joint = osim.SliderJoint("cart", model.getGround(), cart)
model.addJoint(cart_joint)
coord = cart_joint.updCoordinate()
coord.setName("position")


model.addBody(cart)
model.addBody(bar)
model.addBody(sphere)
model.addJoint(pin_joint)
model.addJoint(weld_joint)

# Add actuator
actu = osim.CoordinateActuator()
actu.setCoordinate(coord)
actu.setName("horizontal_force")
actu.setMaxControl(1)
actu.setMinControl(-1)
actu.setOptimalForce(10)

model.addComponent(actu)

model.finalizeConnections()

model.printToXML("./moco/models/inverted_pendulum.osim")

# Create MocoStudy.
# ================
study = osim.MocoStudy()
study.setName("invertedPendulum")

# Define the optimal control problem.
# ===================================
problem = study.updProblem()
problem.setModel(model)
problem.setTimeBounds(0, 20)
problem.setStateInfo("/jointset/pin/pin_coord_0/value", [-10, 10], 0.2, 0)
problem.setStateInfo("/jointset/pin/pin_coord_0/speed", [-50, 50], 0, 0)
problem.setStateInfo("/jointset/cart/position/value", [-2, 2], 0, 0)
problem.setStateInfo("/jointset/cart/position/speed", [-50, 50], 0, 0)

control_goal = osim.MocoControlGoal("effort")
problem.addGoal(control_goal)


solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(15)
solver.set_optim_convergence_tolerance(1e-4)
solver.set_optim_constraint_tolerance(1e-4)

if not os.path.isfile('predictSolution.sto'):
    # Part 1f: Solve! Write the solution to file, and visualize.
    predictSolution = study.solve()
    predictSolution.write('./moco/results/testInvertedPendulum.sto')
    study.visualize(predictSolution)