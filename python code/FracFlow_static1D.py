from dolfin import *
import numpy
import matplotlib.pylab as pylab
#from matplotlib.pylab import *
from scitools.std import linspace, zeros, ones,sqrt
import time

def finite_difference(L, Nx, N, dt, C, P_L, P_R):
    x = linspace(0, L, Nx+1)
    dx = x[1] - x[0]
    C = C*dt/(dx**2)
    print C	
    P   = zeros(Nx+1)
    P_1 = P_R*ones(Nx+1)

    for n in range(0, N):
        # Compute u at inner mesh points
        for i in range(1, Nx):
	    #print P_1
            P[i] = P_1[i] + C*(P_1[i-1]**2 - 2*P_1[i]**2 + P_1[i+1]**2)
	# Insert boundary conditions 
	P[0]=P_L; P[Nx]=P_R
    	# Update u_1 before next step
    	P_1[:]= P
    
    return P, x

tol = 1E-14
def LeftDiricletBoundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol
def RightDiricletBoundary(x, on_boundary):
    return on_boundary and  abs(x[0]-1) < tol


def finite_element_Picard(L, Nx, N, dt, C, P_L, P_R):
    # Define mesh
    mesh = Interval(Nx, 0, L)
    #Define function space and functions
    V = FunctionSpace(mesh, 'Lagrange', 1)
    v = TestFunction(V)
    p = TrialFunction(V)
    p_k = Function(V)
    p_1 = Function(V)
    # Define boundary conditions
    bcl = DirichletBC(V, Constant(P_L), LeftDiricletBoundary)
    bcr = DirichletBC(V, Constant(P_R), RightDiricletBoundary)
    bc = [bcl, bcr]
    # Initial condition
    I_p = Constant(P_R)
    p_1 = interpolate(I_p, V)
    p_k.assign(p_1)
    q = Constant(2*C)
    # Define variational problems
    a = p*v*dx + dt*q*p_k*inner(nabla_grad(p), nabla_grad(v))*dx
    L = p_1*v*dx
    # Time loop
    p = Function(V)     # new unknown function
    tol = 1.0E-5        # tolerance
    maxiter = 25        # max no of iterations allowed
    t = dt
    for n in range(0, N):
        #print 'time =', t
        eps = 1.0           # error measure ||u-u_k||
        iter = 0            # iteration counter
        b = assemble(L)
        while eps > tol and iter < maxiter:
    	    iter += 1
 	    A = assemble(a)
	    bcl.apply(A, b)
	    bcr.apply(A, b) 
	    solve(A, p.vector(), b,'lu')
	    diff = p.vector().array() - p_k.vector().array()
    	    eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    	    p_k.assign(p)
	#print 'iter=%d: norm=%g' % (iter, eps)
        t += dt
        p_1.assign(p_k)
    # return last P
    return p_k.vector()

def finite_element_Newton(L, Nx, N, dt, C, P_L, P_R):
    # Define mesh
    mesh = Interval(Nx, 0, L)
    #Define function space and functions
    V = FunctionSpace(mesh, 'Lagrange', 1)
    v = TestFunction(V)
    dp = TrialFunction(V)
    p_k = Function(V)
    p_1 = Function(V)
    # Define boundary conditions
    bcl = DirichletBC(V, Constant(0), LeftDiricletBoundary)
    bcr = DirichletBC(V, Constant(0), RightDiricletBoundary)
    bc = [bcl, bcr]
    # Initial condition
    I_p = Constant(P_R)
    p_1 = interpolate(I_p, V)
    p_1.vector()[0] = P_L
    p_k.assign(p_1)
    q = Constant(2*C)
    # Define variational problems
    a = dp*v*dx + dt*q*inner(p_k*nabla_grad(dp)+dp*nabla_grad(p_k), nabla_grad(v))*dx
    L = (p_1-p_k)*v*dx - dt*q*inner(p_k*nabla_grad(p_k), nabla_grad(v))*dx
    # Time loop
    dp = Function(V)     # new unknown function
    tol = 1.0E-5        # tolerance
    maxiter = 25        # max no of iterations allowed
    t = dt
    for n in range(0, N):
        #print 'time =', t
        eps = 1.0           # error measure ||u-u_k||
        iter = 0            # iteration counter
        while eps > tol and iter < maxiter:
    	    iter += 1
 	    A = assemble(a)
 	    b = assemble(L)
	    bcl.apply(A, b)
	    bcr.apply(A, b) 
	    solve(A, dp.vector(), b,'lu')
	    diff = dp.vector().array()
    	    eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    	    p_k.vector()[:] = p_k.vector() + dp.vector()
	#print 'iter=%d: norm=%g' % (iter, eps)
        t += dt
        p_1.assign(p_k)
    # return last P
    return p_k.vector()



# Parameters
L = 1
Nx = 20
T = 00
N = 1600
h = 0.05
k = 0.2
P_L = 1
P_R = 0
dt = T/float(N);
DX = L/float(Nx)
C = k*h**2
c = C*dt/DX**2

# Solve by different methods
start = time.clock()
P_fd, x = finite_difference(L, Nx, N, dt, C, P_L, P_R)
print 'FD forward Euler:', time.clock() - start

P_ex = sqrt(1-x)

start = time.clock()
P_Pi = finite_element_Picard(L, Nx, N, dt, C, P_L, P_R)
print 'FE backward Euler Picard:', time.clock() - start

start = time.clock()
P_Ne = finite_element_Newton(L, Nx, N, dt, C, P_L, P_R)
print 'FE bacward Euler Newton:', time.clock() - start


figure(1)
pylab.plot(x, P_ex, x, P_fd, x, P_Pi, x, P_Ne)#, x, P_fe)
pylab.legend(['Exact', 'FD forward Euler', 'FE backward Euler Picard', 'FE backward Euler Newton'])
pylab.xlabel('x')
pylab.ylabel('Pressure')
pylab.title('Constant aperture channel, k*h^2*dt/dx^2 = %.02f, T = %.01f' % (c,T) )
#raw_input()
