import random
from dolfin import *

# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        #values[0] = random.random()
        #values[1] = a*random.random()
        #values[0] = a if abs(x[0]-0.5) < 0.1 else 0.0
        #values[1] = 1.0 if abs(x[0]-0.5) < 0.1 else 0.0
        values[0] = a*exp(-pow((x[0]-0.5)/0.25,2))
	values[1] = exp(-pow((x[0]-0.5)/0.25,2))
    def value_shape(self):
        return (2,)

# Class for interfacing with the Newton solver
class CoupledEquations(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A, reset_sparsity=self.reset_sparsity)
        self.reset_sparsity = False

# Model parameters
k = 5e-1
a = 0.1
b = 1e-2
#c = 1e-1
h_min = 0e-4
dt    = 1.0e-04  # time step
T = 1000*dt
theta = 0      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

print "c = ", k*a**2*dt/0.01**2

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
#mesh = UnitSquare(100, 100)
mesh = UnitInterval(100)
#mesh = UnitCircle(100)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V

# Define trial and test functions
du    = TrialFunction(ME)
q, v  = TestFunctions(ME)

# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dh, dp = split(du)
h,  p  = split(u)
h0, p0 = split(u0)

# Create intial conditions and interpolate
u_init = InitialConditions()
u.interpolate(u_init)
u0.interpolate(u_init)

# Boundary conditions
#def boundary(x, on_boundary):
#    return on_boundary

#bcs = DirichletBC(ME, (Constant(0),Constant(0)), boundary)

# p_(n+theta)
p_mid = (1.0-theta)*p0 + theta*p
h_mid = (1.0-theta)*h0 + theta*h

# Compute conductivity and define variables
h_mid = variable(h_mid)
f = h_mid**3
a = Constant(a)
b = Constant(b)
h_min = Constant(h_min)

# Weak statement of the equations
#L0 = c*h*v*dx - c*h0*v*dx + dt*(h_mid*q*dx - a*p*q*dx + b*inner(nabla_grad(h_mid), nabla_grad(q))*dx)
L0 = h*q*dx - a*p*q*dx + b*inner(nabla_grad(h), nabla_grad(q))*dx
L1 = h*p*v*dx + h_min*p*v*dx - h0*p0*v*dx - h_min*p0*v*dx + dt*k*f*inner(nabla_grad(p_mid*p_mid),nabla_grad(v))*dx
#L1 = h_mid*(p-p0)*v*dx + p_mid*(h-h0)*v*dx + dt*k*f*p_mid*inner(nabla_grad(p_mid),nabla_grad(v))*dx
L = L0 + L1

# Compute directional derivative about u in the direction of du (Jacobian)
a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
problem = CoupledEquations(a, L)
solver = NewtonSolver("lu")
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output file
file = File("output/output.pvd", "compressed")

# Visualization
viz_h = plot(u0.split()[0], ylabel = "h", title = "h")
viz_p = plot(u0.split()[1], ylabel = "p", title = "p")

# Step in time
t = 0.0
it = 0
while (t < T):
    t += dt
    it += 1 
    print "it = ", it, " (t = ", t, "):"
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    file << (u.split()[0], t)
    if it%10==0:
    	viz_h.update(u.split()[0])
    	viz_p.update(u.split()[1])
    #viz_h.write_png("output/Im%04d.png" % it)

viz_h.update(u.split()[0])
viz_p.update(u.split()[1])
interactive()
