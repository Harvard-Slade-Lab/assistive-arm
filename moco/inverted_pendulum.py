import opensim as osim


model = osim.Model()
model.setName("inverted_pendulum")
model.set_gravity(osim.Vec3(0, 0, 0))

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
actu.setOptimalForce(100)

model.addComponent(actu)

model.finalizeConnections()

model.printToXML("inverted_pendulum.osim")

# Create MocoStudy.
# ================
study = osim.MocoStudy()
study.setName("inverted_pendulum")

# Define the optimal control problem.
# ===================================
problem = study.updProblem()

# Model (dynamics).
# -----------------
problem.setModel(model)

# Bounds.
# -------
# Initial time must be 0, final time can be within [0, 5].
problem.setTimeBounds(osim.MocoInitialBounds(0.0), osim.MocoFinalBounds(0.0, 5.0))

# Position must be within [-5, 5] throughout the motion.
# Initial position must be 0, final position must be 0
# problem.setStateInfo(
