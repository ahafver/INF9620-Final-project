from dolfin import *
import numpy
import numexpr as ne

# Time discretizaton
dt = 1e-4
T = 1e-4
# Space discretizaton
nx = 100
DX = 1/float(nx)

# Set coefficients
a = 0.1
b = 1e-3
k = 0.8
P_L = 1.0
P_R = 0.0

h0 = 1e-4

c =  k*a**2*dt/DX**2
print 'c = ', c

# Create mesh and define function space
mesh = UnitInterval(nx)
V = FunctionSpace(mesh, 'Lagrange', 1)

## Define boundary conditions
tol = 1E-14

def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol

bc_hl = DirichletBC(V, Constant(0), left_boundary)
bc_pl = DirichletBC(V, Constant(0), left_boundary)
bc_hr = DirichletBC(V, Constant(0), right_boundary)
bc_pr = DirichletBC(V, Constant(0), right_boundary)

# Set conductivity and capacity 
def q(h):
    return k*h**3

def m(h):
    return h + h0

#Define functions
v = TestFunction(V)
p = TrialFunction(V)
p_k = Function(V)
p_1 = Function(V)
h = TrialFunction(V)
h_k = Function(V)
h_1 = Function(V)

# variational form
a_p = m(h_k)*p*v*dx + dt*q(h_k)*p_k*inner(nabla_grad(p), nabla_grad(v))*dx
L_p = m(h_1)*p_1*v*dx
#a_p = (h_k + Constant(h0))*p*v*dx + Constant(dt*k)*h_k**3*p_k*inner(nabla_grad(p), nabla_grad(v))*dx
#L_p = (h_1 + Constant(h0))*p_1*v*dx
a_h = h*v*dx + Constant(b)*inner(nabla_grad(h), nabla_grad(v))*dx
L_h = Constant(a)*p_k*v*dx

# Initial condition
p = Function(V)     # new unknown function
h = Function(V)     # new unknown function
I_p = Expression('exp(-pow((x[0]-0.5)/0.1,2))')#Constant(P_R)
p_1 = interpolate(I_p, V)
p_k.assign(p_1)

A_h = assemble(a_h) 
b_h = assemble(L_h)
solve(A_h, h_k.vector(), b_h, 'lu')
h_1.assign(h_k) 

#I_h = Expression('0.1*exp(-pow((x[0]-0.5)/0.1,2))')#Constant(P_R)
#h_1 = interpolate(I_h, V)
#h_k.assign(h_1)

# Time loop
tol = 1.0E-5        # tolerance
maxiter = 50        # max no of iterations allowed
t = dt

viz_h = plot(h_k,
             wireframe=False,
             title='Scaled fracture opening',
             #rescale=False,
             axes=True,
             )

viz_p = plot(p_k,
             wireframe=False,
             title='Scaled fracture pressure',
             #rescale=False,
             axes=True,
             )
#interactive()
print 'p_1 =', p_1.vector().array()  
print 'h_1 =', h_1.vector().array()  
#print 'b_p =', b_p.array()

while t <= T:
    print 'time =', t
    eps = 1.0           # error measure ||u-u_k||
    iter = 0            # iteration counter
    b_p = assemble(L_p)
    while eps > tol and iter < maxiter:
    	iter += 1
	  	
        # Solve new opening
	b_h = assemble(L_h)
	solve(A_h, h_k.vector(), b_h, 'lu')

        # Solve new pressure	
	A_p = assemble(a_p)
	bc_pl.apply(A_p, b_p) 	
	bc_pr.apply(A_p, b_p) 	
	solve(A_p, p.vector(), b_p,'lu')

	# Check if converged
	diff = p.vector().array() - p_k.vector().array()
    	eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    	p_k.assign(p)

	print 'iter=%d: norm=%g' % (iter, eps)

    # Plotting
    viz_h.update(h_k)
    viz_p.update(p_k)
    t += dt
    p_1.assign(p_k)
    h_1.assign(h_k)

# Should be at the end

interactive()
