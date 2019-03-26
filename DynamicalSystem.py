"""DynamicalSystem.py: Defines a simple class for a dynamical 
system constructed from two mappings on the unit interval. 
Additional functions for approximating fractal transformations 
constructed via two such dynamical systems are also provided.
Developed and tested using python v2.7.10 and the packages:
numpy v1.15.4, scipy v1.1.0, matplotlib v2.2.3
Brendan Harding, March 2019
"""

__author__    = "Brendan Harding"
__copyright__ = "Copyright 2019"
__license__   = "MIT"
__version__   = "0.1.0"

# Start with imports used in the following classes and functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq #,fsolve,newton,approx_fprime
# Note: I prefer brentq over fsolve, and approx_fprime is only first order
from DeBruijnGenerator import DeBruijnGenerator

class DynamicalSystem:
    """
    A small class to encapsulate general two map overlapping dynamical systems.
    """
    
    def __init__(self,f0,f1,check=False):
        """
        Initiase with f0,f1 (which should be functions [0,1]->R with 
        appropriate restrictions, see check_validity function).
        """
        self.f0 = f0
        self.f1 = f1
        self.rho_max = self.if0(1.0)
        self.rho_min = self.if1(0.0)
        self.rho = 0.5*(self.rho_max+self.rho_min)
        if check: # immediately check the validity
            self.check_validity()
            
    def __call__(self,x):
        return self.f0(x) if x<=self.rho else self.f1(x)
    
    def df0(self,x,eps=1.0E-6):
        #return approx_fprime(self.f0,x,eps) # only first order
        return (self.f0(x+eps)-self.f0(x-eps))/(2.0*eps)
    
    def df1(self,x,eps=1.0E-6):
        #return approx_fprime(self.f1,x,eps) # only first order
        return (self.f1(x+eps)-self.f1(x-eps))/(2.0*eps)
    
    def if0(self,x,tol=1.0E-15): 
        #return fsolve(lambda y:self.f0(y)-x,x0,xtol=tol) 
        #return newton(lambda y:self.f0(y)-x,x0,fprime=self.df0,tol=tol) 
        return brentq(lambda y:self.f0(y)-x,0.0,1.0,xtol=tol)
    
    def if1(self,x,tol=1.0E-15):
        #return fsolve(lambda y:self.f1(y)-x,x0,xtol=tol) 
        #return newton(lambda y:self.f1(y)-x,x0,fprime=self.df1,tol=tol) 
        return brentq(lambda y:self.f1(y)-x,0.0,1.0,xtol=tol)
    
    def check_validity(self,check_derivative=True,verbose=True):
        """
        Checks basic properties to determine if the dynamical system
        is a valid mapping [0,1]->[0,1].
        """
        is_valid = True
        if self.f0(0.0)!=0.0:
            is_valid = False
            if verbose:
                print("DynamicalSystem: Warning: The system does "+
                      "not satisfy f0(0)=0 (f(0)=",self.f0(0.0),")")
        if self.f1(1.0)!=1.0:
            is_valid = False
            if verbose:
                print("DynamicalSystem: Warning: The system does "+
                      "not satisfy f0(1)=1 (f1(1)=",self.f1(1.0),")")
        if self.rho_max<self.rho_min:
            is_valid = False
            if verbose:
                print("DynamicalSystem: Warning: The system does "+
                      "not satisfy rho_max>=rho_min (rho_min,rho_max=",
                      rho_min,rho_max,")")
        if check_derivative:
            # Coarse check if the derivative is bounded below by some d>1
            # Note it is sufficient to check the restricted ranges given
            xs0 = np.linspace(0.0,self.rho_max,129)
            xs1 = np.linspace(self.rho_min,1.0,129)
            d = np.min(np.concatenate((self.df0(xs0),self.df1(xs1))))
            if d<=1.0:
                is_valid = False
                if verbose:
                    print("DynamicalSystem: Warning: The system does "+
                          "not satisfy: d>1 (d<=",d,")")
        return is_valid
    
    def set_rho(self,rho):
        """
        Sets the mask point rho. 
        (A warning is printed if it is not in the valid range.)
        """
        self.rho = rho
        if self.rho>self.rho_max or self.rho<self.rho_min:
            print("DynamicalSystem: Warning: The given rho is outside the "+
                  "range [rho_min,rho_max]=[",rho_min,rho_max,"]")
    
    def get_rho(self):
        """Returns the current value of the mask point rho."""
        return self.rho
    
    def get_rho_range(self):
        """Returns the range of valid values of the mask point rho."""
        return [self.rho_min,self.rho_max]
    
    def plot(self,include_rho=True,show=True,nx=129):
        """Plots the dynamical system."""
        xs = np.linspace(0.0,1.0,nx)
        plt.plot(xs,self.f0(xs),'b-')
        plt.plot([self.rho_max,self.rho_max],[0.0,1.0],'b:')
        plt.plot(xs,self.f1(xs),'r-')
        plt.plot([self.rho_min,self.rho_min],[0.0,1.0],'r:')
        plt.plot([0.0,1.0],[0.0,1.0],'k--')
        if include_rho:
            plt.plot([self.rho,self.rho],[0.0,1.0],'g:')
        plt.xlim(0.0,1.0)
        plt.ylim(0.0,1.0)
        plt.axes().set_aspect(1.0)
        if show:
            plt.show()
    
    def plot_orbit(self,x0,its=10,show=True,nx=129):
        """Plots an orbit of the dynamical system."""
        xs0 = np.linspace(0.0,self.rho,nx)
        plt.plot(xs0,self.f0(xs0),'b-')
        #plt.plot([self.rho_max,self.rho_max],[0.0,1.0],'b:')
        xs1 = np.linspace(self.rho,1.0,nx)
        plt.plot(xs1,self.f1(xs1),'b-')
        #plt.plot([self.rho_min,self.rho_min],[0.0,1.0],'r:')
        plt.plot([0.0,1.0],[0.0,1.0],'k--')
        plt.plot([self.rho,self.rho],[0.0,1.0],'g:')
        dat = np.zeros((2*its+2,2))
        dat[0,0] = x0
        dat[1,:] = x0
        for k in range(its):
            y = self(dat[2*k+1,0])
            dat[2*k+2,0] = dat[2*k+1,0]
            dat[2*k+2,1] = y
            dat[2*k+3,:] = y
        plt.plot(dat[:,0],dat[:,1],'k-',alpha=0.5)
        plt.xlim(0.0,1.0)
        plt.ylim(0.0,1.0)
        plt.axes().set_aspect(1.0)
        if show:
            plt.show()
    
    def plot_inverse(self,show=True,nx=129):
        """Plot the IFS corresponding to the inverse of the dynamical system"""
        xs = np.linspace(0.0,1.0,nx)
        plt.plot(xs,np.vectorize(self.if0)(xs),'b-')
        plt.plot([0.0,1.0],[self.rho_max,self.rho_max],'b:')
        plt.plot(xs,np.vectorize(self.if1)(xs),'r-')
        plt.plot([0.0,1.0],[self.rho_min,self.rho_min],'r:')
        plt.plot([0.0,1.0],[0.0,1.0],'k--')
        #plt.plot([0.0,1.0],[self.rho,self.rho],'g:')
        plt.xlim(0.0,1.0)
        plt.ylim(0.0,1.0)
        plt.axes().set_aspect(1.0)
        if show:
            plt.show()
    
    def tau(self,x,n=50):
        """Map from point in [0,1] into code space via the dynamical system"""
        sigma = np.zeros(n,dtype=np.int8)
        for k in range(n):
            if x<=self.rho:
                sigma[k] = 0
                x = self.f0(x)
            else:
                sigma[k] = 1
                x = self.f1(x)
        return sigma
    
    def tau_plus(self,x,n=50):
        """Map from point in [0,1] into code space via the dynamical system"""
        sigma = np.zeros(n,dtype=np.int8)
        for k in range(n):
            if x>=self.rho:
                sigma[k] = 1
                x = self.f1(x)
            else:
                sigma[k] = 0
                x = self.f0(x)
        return sigma
    
    def pi(self,sigma,y0=0.5):
        """Map from code space to point via the inverse mappings"""
        y = y0
        for k in range(len(sigma)-1,-1,-1):
            if sigma[k]==0:
                y = self.if0(y)
            else:
                y = self.if1(y)
        return y
    
    def get_ergodic_region(self):
        """Return the interval of the ergodic region for the system"""
        return [self.f1(self.rho),self.f0(self.rho)]
    
    def get_alpha_beta(self,n=50):
        """Return the codes bounding the ergodic region for the current rho."""
        return self.tau(self.f0(self.rho),n),self.tau_plus(self.f1(self.rho),n)
    
    def get_alpha_beta_bounds(self,n=50):
        """Return the codes bounding those always in ergodic region."""
        rho_temp = self.rho
        self.rho = self.rho_max
        beta = self.tau_plus(self.f1(self.rho_max),n)
        self.rho = self.rho_min
        alpha = self.tau(self.f0(self.rho_min),n)
        self.rho = rho_temp
        return alpha,beta
    
    def get_symmetric_system(self):
        """Return the dynamcial system which is symmetric to this one."""
        W = DynamicalSystem(lambda x:1.0-self.f1(1.0-x),
                            lambda x:1.0-self.f0(1.0-x))
        W.set_rho(1.0-self.rho)
        return W
    
    def set_symmetric_rho(self,nc=50,delta=1.0E-9,eps=1.0E-12):
        """
        Determine and set the rho which sets up a homeomorphism 
        between two symmetric dynamical systems.
        (This is the algorithm from my AustMS Bulletin paper.)
        """
        
        def distance_to_symmetry(rho):
            rho_temp = self.rho
            self.rho = rho
            sigmaL = self.tau(rho,nc)
            sigmaR = self.tau_plus(rho,nc)
            self.rho = rho_temp
            k = np.argmin(1-sigmaL-sigmaR==0)
            return (1-sigmaL[k]-sigmaR[k])*0.5**k
            
        rhoL = self.rho_min
        rhoR = self.rho_max
        rhoC = 0.5*(rhoL+rhoR)
        distL = distance_to_symmetry(rhoL)
        distR = distance_to_symmetry(rhoR)
        distC = distance_to_symmetry(rhoC)
        while rhoR-rhoL>delta and abs(distC)>eps:
            if distL*distC<0:
                distR = distC
                rhoR = rhoC
            else:
                distL = distC
                rhoL = rhoC
            rhoC = 0.5*(rhoL+rhoR)
            distC = distance_to_symmetry(rhoC)
        self.rho = rhoC
        return self.rho
    
    # end class

def fractalTransformation(F,G,M=256,N=50):
    """
    Computes a fractal transformation between two dynamical systems.
    F,G should be two (valid) DynamicalSystem objects.
    The direct algorithm is used here.
    """
    assert isinstance(F,DynamicalSystem) and isinstance(G,DynamicalSystem)
    assert F.check_validity(True,False) and G.check_validity(True,False)
    x = np.linspace(0.0,1.0,M+1)
    transformation = np.vectorize(lambda x:G.pi(F.tau(x,N)))
    return x,transformation(x)

def fractalTransformationCG(F,G,M=256,N=50,its=16,
                            deBruijn=True,return_Q=False):
    """
    Computes a fractal transformation between two dynamical systems.
    F,G should be two (valid) DynamicalSystem objects.
    The chaos game algorithm is used here.
    If deBruijn is True, its will be interpreted as 2**int(its).
    If deBruijn is False and its<=30, its will be interpreted as int(2**its).
    If deBruijn is False and its> 30, its will be interpreted as int(its).
    """
    assert isinstance(F,DynamicalSystem) and isinstance(G,DynamicalSystem)
    assert F.check_validity(True,False) and G.check_validity(True,False)
    if deBruijn:
        its = int(its)
        if its>32:
            print("fractalTransformationCG: Warning: A very long sequence "+
                  "length has been requested! (2**",its,")")
    else:
        if its<=30:
            its = int(2.0**its)
        else:
            its = int(its)
    rho = F.get_rho()
    tau_L = F.tau(rho,N+1)
    tau_R = F.tau_plus(rho,N+1)
    sigma = np.zeros(N+1,dtype=np.int8)
    X = np.linspace(0.0,1.0,M+1)
    H = X.copy()
    Q = np.zeros(M+1,dtype=np.int)
    Q[0],Q[M] = N,N # since the end points are always correct
    q,x,y = 0,1.0,1.0
    def address_distance(alpha,beta):
        k = np.argmin(alpha==beta)
        return (beta[k]-alpha[k])*0.5**k
    if deBruijn:
        db_2 = DeBruijnGenerator(2,its)
        #for _ in range(db_2.length()): # beware of overflow!
        while not db_2.is_complete(): # this is better
            sigma = np.roll(sigma,1)
            sigma[0] = db_2()
            if sigma[0]==0:
                x = F.if0(x)
                y = G.if0(y)
            else:
                x = F.if1(x)
                y = G.if1(y)
            if sigma[0]==0:
                if address_distance(sigma,tau_L)<0:
                    q = 0
            else:
                if address_distance(tau_R,sigma)<0:
                    q = 0
            k = int(0.5+x*M)
            # Should really check k is in the right range (i.e. 0,1,...,M)
            # but this shouldn't happen and is somewhat expensive to check
            if Q[k] < q:
                H[k] = y
                Q[k] = q
            q += 1
        # end while
    else:
        for _ in range(its):
            sigma = np.roll(sigma,1)
            sigma[0] = np.random.randint(2)
            if sigma[0]==0:
                x = F.if0(x)
                y = G.if0(y)
            else:
                x = F.if1(x)
                y = G.if1(y)
            if sigma[0]==0:
                if address_distance(sigma,tau_L)<0:
                    q = 0
            else:
                if address_distance(tau_R,sigma)<0:
                    q = 0
            k = int(0.5+x*M)
            # Should really check k is in the right range (i.e. 0,1,...,M)
            # but this shouldn't happen and is somewhat expensive to check
            if Q[k] < q:
                H[k] = y
                Q[k] = q
            q += 1
        # end for
    # end if/else
    if return_Q:
        return X,H,Q
    return X,H

if __name__=="__main__":
	# If run as a script, perform a simple test of an affine example
	affineDS = DynamicalSystem(lambda x:1.5*x,lambda x:(5.0*x-2.0)/3.0)
	assert np.abs(affineDS.set_symmetric_rho()-0.554383369)<1.0E-9
	print("DynamicalSystem.py: all tests passed!")
