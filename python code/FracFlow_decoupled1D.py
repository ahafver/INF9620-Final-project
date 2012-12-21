from dolfin import *
import numpy
import matplotlib.pylab as pylab
#from matplotlib.pylab import *
from scitools.std import linspace, zeros, ones,sqrt
import time

def finite_difference(L, Nx, N, dt, C, P_L, P_R):
    x = linspace(0, L, Nx+1)
    dx = x[1] - x[0]
    C = 0.4*C*dt/(dx**2)
    print C	
    Q   = zeros(Nx+1)
    Q_1 = P_R**2*ones(Nx+1)

    for n in range(0, N):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            Q[i] = Q_1[i] + C*(Q_1[i-1]**2.5 - 2*Q_1[i]**2.5 + Q_1[i+1]**2.5)
	# Insert boundary conditions 
	Q[0]=P_L**2; Q[Nx]=P_R**2
    	# Update u_1 before next step
    	Q_1[:]= Q
    
    return sqrt(Q), x

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
    q = TrialFunction(V)
    q_k = Function(V)
    q_1 = Function(V)
    # Define boundary conditions
    bcl = DirichletBC(V, Constant(P_L**2), LeftDiricletBoundary)
    bcr = DirichletBC(V, Constant(P_R**2), RightDiricletBoundary)
    bc = [bcl, bcr]
    # Initial condition
    I_q = Constant(P_R**2)
    q_1 = interpolate(I_q, V)
    q_k.assign(q_1)
    c = Constant(C)
    # Define variational problems
    a = q*v*dx + dt*c*q_k**1.5*inner(nabla_grad(q), nabla_grad(v))*dx
    L = q_1*v*dx
    # Time loop
    q = Function(V)     # new unknown function
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
	    solve(A, q.vector(), b,'lu')
	    diff = q.vector().array() - q_k.vector().array()
    	    eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    	    q_k.assign(q)
	#print 'iter=%d: norm=%g' % (iter, eps)
        t += dt
        q_1.assign(q_k)
    # return last P
    return sqrt(q_k.vector().array())

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
    a = 2*p_k*dp*v*dx + dt*q*inner(p_k**4*nabla_grad(dp)+4*dp*nabla_grad(p_k), nabla_grad(v))*dx
    L = (p_1**2-p_k**2)*v*dx - dt*q*inner(p_k**4*nabla_grad(p_k), nabla_grad(v))*dx
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
Nx = 50
T = 1
N = 100
a = 0.1
k = 0.8
P_L = 1
P_R = 0
dt = T/float(N);
DX = L/float(Nx)
C = k*a**2
c = C*dt/DX**2

# Solve by different methods
start = time.clock()
P_fd, x = finite_difference(L, Nx, N, dt, C, P_L, P_R)
print 'FD forward Euler:', time.clock() - start

P_ex = (1-x)**0.2

#start = time.clock()
#P_Pi = finite_element_Picard(L, Nx, N, dt, C, P_L, P_R)
#print 'FE backward Euler Picard:', time.clock() - start

#start = time.clock()
#P_Ne = finite_element_Newton(L, Nx, N, dt, C, P_L, P_R)
#print 'FE bacward Euler Newton:', time.clock() - start


figure(1)
pylab.plot(x, P_ex, x, P_fd)#, x, P_Pi, x, P_Ne)#, x, P_fe)
pylab.legend(['Exact', 'FD forward Euler'])#, 'FE backward Euler Picard', 'FE backward Euler Newton'])
pylab.xlabel('x')
pylab.ylabel('Pressure')
pylab.title('Decoupled problem, k*a^2*dt/dx^2 = %.02f, T = %.01f' % (c,T) )
raw_input()
