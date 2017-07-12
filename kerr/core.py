
# Import underlying basic modules
from kerr.basics import *
# import dill

'''
Method to load tabulated QNM data, interpolate and then output for input final spin
'''
def leaver( jf,                     # Dimensionless BH Spin
            l,                      # Polar Index
            m,                      # Azimuthal index
            n =  0,                 # Overtone Number
            p = None,               # Parity Number for explicit selection of prograde (p=1) or retrograde (p=-1) solutions.
            s = -2,                 # Spin weight
            return_splines=False,   # Toggel to return splines rather than values
            Mf = 1.0,               # BH mass. NOTE that the default value of 1 is consistent with the tabulated data. (Geometric units ~ M_bare / M_ADM )
            verbose = False ):      # Toggle to be verbose

    # Import useful things
    import os
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from numpy import loadtxt,exp,sign,abs
    from numpy.linalg import norm

    # Validate jf input: case of int given, make float. NOTE that there is further validation below.
    if isinstance(jf,int): jf = float(jf)
    # Valudate s input
    if abs(s) != 2: raise ValueError('This function currently handles on cases with |s|=2, but s=%i was given.'%s)
    # Validate l input
    # Validate m input

    #%%%%%%%%%%%%%%%%%%%%%%%%%# NEGATIVE SPIN HANDLING #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Define a parity value to be used later:
    # NOTE that is p>0, then the corotating branch will be loaded, else if p<0, then the counter-rotating solution branch will be loaded.
    if p is None:
        p = sign(jf) + int( jf==0 )
    # NOTE that the norm of the spin input will be used to interpolate the data as the numerical data was mapped according to jf>=0
    # Given l,m,n,sign(jf) create a RELATIVE file string from which to load the data
    cmd = parent(  parent(os.path.realpath(__file__))  )
    #********************************************************************************#
    m_label = 'm%i'%abs(m) if (p>=0) or (abs(m)==0) else 'mm%i'%abs(m)
    #********************************************************************************#
    data_location = '%s/bin/data/kerr_qnm_london/l%i/n%il%i%s.dat' % (cmd,l,n,l,m_label)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    # Validate data location
    if not os.path.isfile(data_location): raise ValueError('The OS reports that "%s" is not a file or does not exist. Either the input QNM data is out of bounds (not currently stored in this repo), or there was an input error by the user.' % green(data_location) )

    # Load the QNM data
    data = loadtxt( data_location )

    # Extract spin, frequencies and separation constants
    JF = data[:,0]
    CW = data[:,1] - 1j*data[:,2] # NOTE: The minus sign here sets a phase convention
                                  # where exp(+1j*cw*t) rather than exp(-1j*cw*t)
    CS = data[:,3] - 1j*data[:,4] # NOTE: There is a minus sign here to be consistent with the line above

    # Validate the jf input
    njf = norm(jf) # NOTE that the calculations were done using the jf>=0 convention
    if njf<min(JF) or njf>max(JF):
        warning('The input value of |jf|=%1.4f is outside the domain of numerical values [%1.4f,%1.4f]. Note that the tabulated values were computed on jf>0.' % (njf,min(JF),max(JF)) )

    # Here we rescale to a unit mass. This is needed because leaver's convention was used to perform the initial calculations.
    M_leaver = 0.5
    CW *= M_leaver

    # Interpolate/Extrapolate to estimate outputs
    cw = spline( JF, CW.real )(njf) + 1j*spline( JF, CW.imag )( njf )
    cs = spline( JF, CS.real )(njf) + 1j*spline( JF, CS.imag )( njf )

    # If needed, use symmetry relationships to get correct output.
    def qnmflip(CW,CS):
        return -cw.conj(),cs.conj()
    if m<0:
        cw,cs =  qnmflip(cw,cs)
    if p<0:
        cw,cs =  qnmflip(cw,cs)

    # NOTE that the signs must be flipped one last time so that output is
    # directly consistent with the argmin of leaver's equations at the requested spin values
    cw = cw.conj()
    cs = cs.conj()

    # Here we scale the frequency by the BH mass according to the optional Mf input
    cw /= Mf

    #
    return cw,cs

'''
Useful Method for estimating QNM locations in leaver solution space: Estimate the
local minima of a 2D array
'''
def localmins(arr,edge_ignore=False):

    import numpy as np
    import scipy.ndimage.filters as filters
    import scipy.ndimage.morphology as morphology

    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    __localmin__ =  list( np.where(detected_minima) )

    # Option: Ignore mins on domain boundaries
    if edge_ignore:
        isonedge0 =  lambda x: (x==0) or (x==(len(arr[:,0])-1))
        isonedge1 =  lambda x: (x==0) or (x==(len(arr[0,:])-1))
        mask = np.ones( __localmin__[0].shape, dtype=bool )
        for k in range(len(__localmin__[0])):
            mask[k] = not ( isonedge0( __localmin__[0][k] ) or isonedge1( __localmin__[1][k] ) )
        __localmin__[0] = __localmin__[0][mask]
        __localmin__[1] = __localmin__[1][mask]

    #
    return __localmin__


'''
Given data set xx yy constrcut an interpolating polynomial that passes through all points (xx,yy). The output is a function object.
http://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python
'''
def newtonpoly(xx,yy):

    import numpy as np
    #import matplotlib.pyplot as plt

    def coef(x, y):
        '''x : array of data points
           y : array of f(x)  '''
        x.astype(float)
        y.astype(float)
        n = len(x)
        a = []
        for i in range(n):
            a.append(y[i])

        for j in range(1, n):

            for i in range(n-1, j-1, -1):
                a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

        return np.array(a) # return an array of coefficient

    def Eval(a, x, r):

        ''' a : array returned by function coef()
            x : array of data points
            r : the node to interpolate at  '''

        x.astype(float)
        n = len( a ) - 1
        temp = a[n]
        for i in range( n - 1, -1, -1 ):
            temp = temp * ( r - x[i] ) + a[i]
        return temp # return the y_value interpolation

    #
    A = coef(xx,yy)
    return lambda r: Eval( A, xx, r )


# Calculate the inner product between exp(1i*w_i) and exp(1i*cw_j)
# translated from MATLAB (in DAKIT::y_depict.m)
def expprod( w_i, cw_j, T1, T2, m ):
    #
    # Analytically evaluate the integreal:
    #
    # Int_T1 ^ T2 ( exp(1j*cw_j*t) exp(-1j*w_i*t) dt )
    #
    # NOTE: w_i should be real and cw_j should be complex for the use of QNM frequecies

    # Import useful things
    from numpy import exp

    # Enforce that the imaginary part of cw_j is positive as the analytic form of the integral was calculated under this assumption. NOTE that the sign of the imag(cw_j) changes under a phase or fourier transform convention when handling Teukolsky's equations.
    if cw_j.imag < 0: cw_j = cw_j.conj()

    # Analytically evaluate the desired integral
    if 0 != m:
        #
        sigma_ij = cw_j - w_i
        x = -1j * ( exp(1j*sigma_ij*T2) - exp(1j*sigma_ij*T1) ) / sigma_ij
    else:
        x = 0
        for s in [1,-1]:
            #
            sigma_ij = cw_j-s*w_i
            x = x + -1j * ( exp(1j*sigma_ij*T2) - exp(1j*sigma_ij*T1) ) / sigma_ij
        #
        x = 0.5*x

    #
    return x


# Low level algorithm to find QNM amplitues via scattering matrix approach
def nodd_helper( Y,                 # Waveform range
                 t,                 # waveform domain
                 wvals,             # frequencies for QNMs
                 gwf_m,             # m of multipolar set
                 qnmfit_object=None,# optional ralted qnmfit object
                 real_only=False):  # toggle for real only data (NOTE that this behavior should be depreciated for a system that uses the input range to determine datatype)
    '''
    Low level algorithm to find QNM amplitues via scattering matrix approach
    '''

    #
    from numpy import array
    # beta,complex_w = lsexp( Y, t, wvals )

    if not False:
        #
        from numpy import exp,trapz,zeros,array,amax,dot,std,complex64
        from numpy.linalg import inv,cond

        #
        N = len(wvals)

        #
        tiny = 1e-20

        #
        scale_fun = lambda p: wvals[p]**-1

        # Define "projection space"
        # --------------------------------------------- %
        f = lambda p,X: exp( 1j * wvals[p].real * X )

        #
        prod = lambda A,B: trapz( A * B.conj(), t )
        alpha = ydcmp(t,Y,N=N,fun=f,prod=prod)

        # #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%# #
        #            HEART OF THE NODD ALGORITHM            #
        # #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%# #

        U = zeros( (len(wvals),len(wvals)) , dtype=complex )
        T1,T2 = t[0],t[-1]
        complex_w = array( wvals )
        for i in range(N):
            w_i = complex_w[i].real
            for j in range(N):
                cw_j = complex_w[j]
                U[i,j] = expprod( w_i, cw_j, T1, T2, gwf_m * (not real_only) )
                U[i,j] /= scale_fun(j)

        # rescale U
        Umax = amax(U)
        U /= Umax

        # Calculate the spectrum in the test basis: alpha = U*beta --> beta = U\alpha
        rcond = 1.0/cond(U)
        if rcond > tiny:
            beta = dot( inv(U) , alpha )
            beta /= Umax # account for the rescaling of U
        else: # IGNORE singular matricies
            beta = 0*alpha

        # Calculate output amplitudes (e.g. rescale such that, effectively, scale_fun=1)
        for j in range(len(beta)):
            beta[j] = beta[j] / scale_fun(j)

    #
    return beta,complex_w



# Wrapper for nodd_helper that takes in a gwf object (nrutils) as the first input
def nodd_helper_gwfo(gwf_object,wvals,qnmfit_object=None,real_only=False):

    # If the noise floor is specified, only use data
    # up to the noise floor.
    Y = gwf_object.y
    t = gwf_object.t

    # Apply the main routine
    beta,complex_w = nodd_helper(Y,t,wvals,gwf_object.m,qnmfit_object=qnmfit_object,real_only=real_only)

    #
    return beta,complex_w


# Given a set of bsais functions (fun), an inner-product (prod), and some data (t,Y), decompose the data into the moments defined by prod( fun(k,t), Y(t) ), where k labels the k-th basis function
def ydcmp( t,                   # Domain
           Y,                   # Range
           N = 10,              # Number of moments to calculate
           fun = None,          # Descrete basis function to use
           prod = None,         # Inner-product operation (function)
           verbose=False ):
    #
    from numpy import trapz,zeros

    #
    if prod is None:
        prod = lambda A,B: trapz( A * B.conj(), t )

    #
    alpha = zeros( (N,), dtype=complex );
    for k in range(N):
        P = fun(k,t)
        alpha[k] = prod( Y, P )

    #
    return alpha


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Class for QNM Objects, and lists of qnm objects '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
class qnmobj:
    # Initialize the object
    def __init__(this,
                 Mf,
                 jf,
                 z,                 # array of qnm indeces, OR list of arrays
                 verbose=False):

        # Import needed things
        from kerr import leaver
        from numpy import array

        # ----------------------------------------- #
        #              Validate z input             #
        # ----------------------------------------- #
        if isinstance(z[0],(list,tuple)):
            # a list of QNM coordinates are given
            # validate the list
            for a in z:
                if not isinstance(a,(list,tuple)):
                    msg = 'qnm coordinate must be list or tuple; check z input: z = %s'%list(z)
                    error(msg)
                else:
                    if len(a)/4 != int(len(a)/4):
                        msg = 'list of qnm indeces found to have improperly formatted element, z = %s'%list(a)
                        error(msg)
        elif isinstance(z[0],(int,float)):
            # a single qnm is given
            if len(z)/4 != int(len(z)/4):
                msg = 'list of qnm indeces found to have improperly formatted element, z = %s'%list(z)
                error(msg)
            # put it in a list of length one
            z = [z]

        # Assign basic class properties from inputs
        this.Mf,this.xf,this.verbose,this.z = Mf,jf,verbose,z

        #
        if this.verbose:
            msg = 'Generating qnmobj\'s for z in:'
            alert(msg)
            for k in this.z:
                msg = '\t%s'%list(k)
                alert(msg)

        # store a list of the current objects orders
        this.order = []
        for k in z:
            this.order.append( len(k) )

        # Get QNM Frequency and Separation Constant, and store to lists
        # Also create maps between l,m,n,p and QNM info
        this.cw, this.sc = [],[]
        this.cwmap, this.icwmap = {},{}
        this.scmap, this.iscmap = {},{}
        for k in this.z:
            if len(k) == 4:
                l,m,n,p = k
                cw,sc = leaver(jf,l,m,n,p,Mf=this.Mf)
                this.cw.append( cw )        # append list of qnm frequencies
                this.sc.append( sc )        # append list of sep consts
                this.cwmap[k] = cw          # update dict between k and cw
                this.icwmap[cw] = k         # update dict between cw and k
                this.scmap[k] = sc          # update dict between k and sc
                this.iscmap[sc] = k         # update dict between sc and k
            elif len(k) == 8:
                l1,m1,n1,p1 = k[:4]
                l2,m2,n2,p2 = k[4:]
                cw1,sc1 = leaver(jf,l1,m1,n1,p1,Mf=this.Mf)
                cw2,sc2 = leaver(jf,l2,m2,n2,p2,Mf=this.Mf)
                cw = cw1 + cw2
                sc = sc1*sc2
                this.cw.append( cw )        # append list of qnm frequencies
                this.sc.append( sc )        # append list of sep consts
                this.cwmap[k] = cw          # update dict between k and cw
                this.icwmap[cw] = k         # update dict between cw and k
                this.scmap[k] = sc          # update dict between k and sc
                this.iscmap[sc] = k         # update dict between sc and k
            else:
                msg = 'only QNM orders up to 2 are handled, and note that the frequencies and separation constatns are guesses for order 2; there is no formal theory yet!'
                error(msg)

    # Return the spheriodal harmonic at theta dn phi for this QNM
    def slm(this,theta,phi):
        #
        from kerr import slm as __slm__
        #
        s = []
        for k in this.z:
            if len(k) == 4:
                l,m,n,p = k
                s.append(  __slm__( this.xf*p,l,m*p,n,theta,phi )  )
            elif len(k) == 8:
                l1,m1,n1,p1 = k[:5]
                l2,m2,n2,p2 = k[5:]
                s.append(  __slm__( this.xf*p1,l1,m*p1,n1,theta,phi ) * __slm__( this.xf*p2,l2,m2*p2,n2,theta,phi )  )
            else:
                msg = 'only QNM orders up to 2 are handled, and note that the frequencies and separation constatns are guesses for order 2; there is no formal theory yet'
                error(msg)
        #
        return s


# Class for QNM fitting process
class qnmfit:

    #
    nrange = [0,1]
    prange = [1,-1] # [1,-1]

    #
    def __init__( this,                 # The current object
                  gwfo,                 # ringdown only gwf object from nrutils
                  refz = None,          # reference list of QNM coordinated; if not input, relevant qnm coordinates will be guessed from the gwfo ll and mm values
                  prange = None,        # Parity labels to consider
                  nrange = None,        # Overtone labels to consider
                  greedy = False,       # Toggle for greedy fitting. NOTE: a negative greedy algorithm is used.
                  Mfxf = None,          # If one does not trust the run's metadata, then manually input the final mass and spin to use; otherwise, the metadata values will be taken; Mfxf = (Mf,xf)
                  verbose = False):     # Let the people know

        #
        from numpy import arange,array,sign,mod,std

        # NOTE that the order of the methods below matters significantly

        #
        this.verbose = verbose
        # Store label ranges to use in __z__() method
        # NOTE that the default values are determined in __z__() as well
        this.prange, this.nrange = prange, nrange

        # Validate first input
        if gwfo.__class__.__name__ != 'gwf':
            msg = 'first input must be gwf object from nrutils package corresponding to a spherical multipole time series after the gwylm.ringdown(...) method has been applied; instead a %s was found'%yellow(type(gwfo).__name__)
            error(msg)

        # Store needed information from gwfo and generate qnm information
        this.setfields( gwfo, refz, Mfxf=Mfxf )

        # Fit the input data using the desired method
        if greedy:
            this.__greedyfit__()
        else:
            this.__fit__()

    # Generate list of QNM coordinates for this object and store to the current object
    def __z__(this):

        #
        from numpy import arange,array,sign,mod

        # Given ll and mm, determine the spheroidal indeces that are relevant
        lmin = max( this.ll-1, 2 )
        lmax = this.ll if this.prange is this.spin_prange else this.ll+1
        # lmax = this.ll if this.prange is this.spin_prange else this.ll+1
        lrange = arange( lmin,lmax+1 )
        mrange = array([this.mm])

        # First Order Modes
        z = [ this.homez ]
        for p in this.prange:
            for l in lrange:
                for m in mrange[abs(mrange)<=l]:
                    for n in this.nrange:
                        z.append( (l,m,n,int(p)) )

        # Second Order Modes: NOTE that only slect modes will be allowed
        m1range = range(1,lmax+1)
        m2range = range(1,lmax+1)
        for p in [int(sign(this.xf))]:
            for l1 in range( 2, max(2,this.ll-1) ):
                for l2 in [l1,l1+1]:
                    for n in [0]:
                        for m1 in m1range:
                            for m2 in [l2,l2-1]:
                                meff = (m1+m2)*sign(this.mm)
                                mrule = (m2<=l2) and (m1<=l1)
                                if mod(this.mm,2):
                                    mrule = mrule and ( l1==m1 )
                                else:
                                    mrule = mrule and ( l1==m1 ) and ( l2==m2 )
                                if (meff==this.mm) and mrule :
                                    z.append( (l1,m1*sign(this.mm),n,p,l2,m2*sign(this.mm),n,p) )

        # NOTE that in the case where the decomposition is in a fixed frame, and the remnant BH has a significant recoil, and/or experiences significant center of mass motion throughout it's evolution, there can be secular mixing of QNMs with different m eigenvalues. Here we consider adding these modes to the fit.
        # ref: https://arxiv.org/pdf/1509.00862.pdf
        # ref: https://arxiv.org/pdf/0805.1017.pdf

        # Add the lp(p for prime) = mp = m-1 fundamental mode
        lp = max(2,abs(this.mm)-1)
        mp = sign(this.mm)*(abs(this.mm)-1)
        lp2 = this.ll
        mp2 = sign(this.mm)*(this.ll)
        if abs(this.xf)>0.1: # Spins below this threshold will be considered to have nearly degenerate overlaps with the secular modes below
            if mp!=0: z.append( ( lp,  mp,  0, this.prange[0] ) )
            z.append( ( lp2, mp2, 0, this.prange[0] ) )
            z.append( ( lp2+1, mp2+sign(this.mm)*1, 0, this.prange[0] ) )
            z.append( ( lp2+3, mp2+sign(this.mm)*3, 0, this.prange[0] ) )
        else:
            z.append( ( lp2+3, mp2+sign(this.mm)*3, 0, this.prange[0] ) )


        #
        this.z = z

    # Store needed information from gwfo and generate qnm information
    def setfields(this,gwfo,refz=None,Mfxf=None):

        #
        from numpy import sign
        from numpy.linalg import norm

        # Validate first input
        if gwfo.__class__.__name__ != 'gwf':
            msg = 'first input must be gwf object from nrutils package corresponding to a spherical multipole time series'
            error(msg)

        #
        this.gwfo = gwfo

        # NOTE that the gwfo must be a spherical multipole moment
        this.ll,this.mm = gwfo.l,gwfo.m

        # NOTE that nrutils uses xf to mean the same as the "jf" that may be used elsewhere
        if Mfxf is None:
            # Take from metadata
            this.Mf = gwfo.mf/(gwfo.m1+gwfo.m2)
            this.xf = gwfo.ref_scentry.xf
        else:
            # from input,assuming correct format: Mfxf = (Mf,xf)
            this.Mf,this.xf = Mfxf

        #
        if this.nrange is None:
            this.nrange = [0,1]

        # Store a p range determined by the spin sign relative the initial L direction
        this.spin_prange = [ int(sign(this.xf)) ]
        # Store the p range to use for fitting depending on the prange value input into the constructor
        if this.prange is None:
            this.prange = this.spin_prange

        # Store the QNM index that is expected to be the most dominant in this multipole
        this.homez = ( this.ll, this.mm, 0, int(sign(this.xf)) )

        # Generate QNM coordinats, OR load QNM coordinates from reference object
        if refz is None:
            this.__z__()
        else:
            this.z = refz

        # Ensure that unique z are given
        this.z = list( set(this.z) )

        # Generate QNM objects
        this.qnminfo = qnmobj( this.Mf, this.xf, this.z, verbose = this.verbose )

    # Use a greedy algorithm to fit the data. NOTE that the method below effectively decorates the __fit__ method
    def __greedyfit__(this,plot=False,show=False,fitatol=5e-5):
        '''
        = Compute a fit of the current object's gwf under the following program:

         * Let the largest possible set of QNM applicable be defined by this.__z__()
         * A greedy algorithm will be applied to take away QNM from this set such that the fit becomes better, or such that the fit becomes worse by some small amount
         * Then, the fit is performed once more, but statistics are estimated by slowly shrinking the fitting region to the right.
         * The user should verify that all remaining QNMs display flat amplitudes when this.plot_beta_arr() is executed.

        == One could argue that:

        A. Statistics should be used within the action of the greedy algorithm. While this may be true in pricnciple, here we find that the statistically "best" estimator is always close or equal to the one reported by __fit__ (wrather than __statsfit__). In other words, the frmse is best when the fitting region is largest. This suggests that computtional effort need not be expended by computing statistics at every greedy step.

        B. ?

        '''

        #
        from numpy import array,exp,sum,arange,trapz
        from copy import deepcopy as copy

        # Initiate the fit using all QNM from this.__z__()
        this.__fit__()

        # Create a lexicon of symbols to consider for model learning
        bulk = this.z

        # Prepare inputs for generalized positive greedy algorithm
        def action(trial_boundary):
            trial_z = trial_boundary
            trial_cw = [ this.qnminfo.cwmap[z] for z in trial_z ]
            this.__fit__( wvals = trial_cw )
            # this.__statsfit__( wvals = trial_cw )
            estimator = this.frmse
            return estimator,this
        def plotqnmfit(that): that.plot()

        # Only use the ngative greedy process to remove unneeded QNMs
        boundary = bulk
        est_list = [this.frmse+0.01,this.frmse]
        # Apply a negative greedy process to futher refine the symbol content
        B = ngreedy( boundary, action, plot = plot, show=show, plotfun = plotqnmfit, verbose = this.verbose, ref_est_list = est_list, permanent=[this.homez] )

        # Compute fit with statistics
        this.__statsfit__( wvals = this.__cw__ )


    # Fit the gwf at a single T0, no statistics
    def __fit__(this,gwfo=None,wvals=None):

        #
        from numpy import array,exp,sum,arange,trapz,std

        # Hande gwf object input (NOTE that sefields transfers info from the gwfo to the current object)
        if not ( gwfo is None ):
            this.setfields( gwfo )
        else:
            gwfo = this.gwfo

        #
        wvals = this.qnminfo.cw if wvals is None else wvals
        beta,cw = nodd_helper_gwfo(gwfo,wvals,this)

        # Sort amplitudes and frequencies by norm of amplitudes
        _map = arange( len(beta) )
        _map = sorted( _map, key = lambda p: abs(beta[p]), reverse=True )
        beta = beta[_map]
        cw = cw[_map]

        #
        this.__a__ = beta
        this.__cw__ = cw
        #
        this.amap,this.iamap = {},{}
        for k,_cw in enumerate(this.__cw__):
            _z = this.qnminfo.icwmap[ _cw ]
            this.amap[ this.__a__[k] ] = _z
            this.iamap[ _z ] = this.__a__[k]

        # Calculate colors for plotting
        weights = [ trapz(abs(a)*exp(-gwfo.t*abs(cw[k].imag)),gwfo.t) for k,a in enumerate(beta) ]
        this.__clr__ = rgb( len(weights), jet=True, weights=weights, reverse=not True )

        # Calculate and store the fit residual
        y = this.gwfo
        g = this.feval( y.t, gwfout=True )
        this.frmse = abs( std( g.y - y.y ) / std(y.y) )


    # Fit the gwf at multiple T0 and report moments
    def __statsfit__(this,gwfo=None,wvals=None):

        #
        from numpy import array,exp,sum,arange,trapz,std,vstack,amin,amax,mean,unwrap,angle,where,median,sqrt,argmin

        # Hande gwf object input (NOTE that sefields transfers info from the gwfo to the current object)
        if not ( gwfo is None ):
            this.setfields( gwfo )
        else:
            gwfo = this.gwfo

        # Define starting inputs for nodd
        wvals = this.qnminfo.cw if wvals is None else wvals
        Y,t = gwfo.y,gwfo.t

        # Define the maximum to the right of t[0] through which statistics will be estimated
        this.max_t_shift = 10 # (M)

        # Define sub region in time over which to estimate statistics
        maxdex = where( t>(t[0]+this.max_t_shift) )[0][0]
        index_range = range(maxdex+1) # NOTE that this line is prone to bugs for short data sets
        beta_arr_list,cw = [],array(wvals)
        frmse_list,shift_list = [],[]
        for k in index_range:
            # Get the fit amplitudes for this shift
            beta_k,_ = nodd_helper(Y[k:],t[k:]-t[k],wvals,gwfo.m,this)
            # Rotate the amplitudes so that they are relative to the initial starting time
            time_shift = -t[k]
            beta_k *= exp( 1j * cw.conj() * time_shift )
            # The effect is something with phase: exp( 1j * cw * ( t[k] + t[0] - t[k] ) )
            # NOTE that all of the cw are the same in content: cw_k = cw_k'
            # Add these amplitudes to the holder
            beta_arr_list.append( list(beta_k) )
            # Calculate and store the fit residual
            y = this.gwfo
            g = this.feval( y.t, gwfout=True, a=beta_k, cw=cw )
            frmse_k = abs( std( g.y - y.y ) / std(y.y) )
            frmse_list.append( frmse_k )
            shift_list.append( t[k] )

        # Convert the beta_arr to a numpy array
        beta_arr = vstack( beta_arr_list ).T

        # NOTE that we will label modes whose ampliutdes vary by 50% or more as untrustworthy
        amps = abs(beta_arr)
        # Calculate percent changes by comparing min and max vals to median
        quality = 1 - abs( amax(amps,1)-amin(amps,1) ) / median( amps, 1 )
        this.trust_mask = quality > 0.50
        # print beta_arr.shape
        # beta_arr = beta_arr[ mask,: ]
        # cw = cw[mask]
        # print beta_arr.shape


        # NOTE that columns are labeled by time_shift, rows by frequncy
        # Compute moments in time_shift. The "1" in the mean input signals to average over columns
        beta_mean = mean( beta_arr, 1 )
        beta_median = median( beta_arr, 1 )
        # beta_mean = mean( abs(beta_arr), 1 ) * exp(  1j * mean( unwrap(angle(beta_arr),1) )  )
        beta_std  = std(  beta_arr, 1 )
        beta_min  = amin( beta_arr, 1 )
        beta_max  = amax( beta_arr, 1 )
        beta_range= beta_max-beta_min

        # Store stat information
        alert('Storing median QNM Amplitudes.')
        this.__a__ = beta_median
        this.__a_mean__ = beta_mean
        this.__a_std__ = beta_std
        this.__a_min__ = beta_min
        this.__a_max__ = beta_max
        this.__a_range__ = beta_range
        this.__a_arr__ = beta_arr
        this.__cw__ = cw
        #
        this.__frmse_arr__ = array(frmse_list)
        this.__shift_arr__ = array(shift_list)

        #
        this.amap,this.iamap = {},{}
        this.astdmap,this.iastdmap = {},{}
        for k,_cw in enumerate(this.__cw__):
            #
            _z = this.qnminfo.icwmap[ _cw ]
            #
            this.amap[ this.__a__[k] ] = _z
            this.iamap[ _z ] = this.__a__[k]
            #
            this.astdmap[ this.__a_std__[k] ] = _z
            this.iastdmap[ _z ] = this.__a_std__[k]

        # Calculate and store the fit residual
        y = this.gwfo
        g = this.feval( y.t, gwfout=True )
        this.frmse = abs( std( g.y - y.y ) / std(y.y) )
        # this.frmse = min( frmse_list )

        # Calculate colors for plotting
        pwr = sqrt( trapz( g.y*g.y.conj(), g.t ) )
        prod = lambda A,B: abs(trapz(  A.conj()*B , g.t )/pwr)
        weights = [ prod( g.y-a*exp(1j*cw[k].conj()*g.t), g.t ) for k,a in enumerate(this.__a__) ]
        # weights = [ trapz(abs(a)*exp(-gwfo.t*abs(cw[k].imag)),gwfo.t) for k,a in enumerate(this.__a__) ]
        this.__clr__ = rgb( len(weights), jet=True, weights=weights, reverse=True )

        # plot_beta_arr()
        # raise

    def plot_beta_arr(this):

        from matplotlib.pyplot import figure,plot,show,gca,axhline,subplot,legend,xlim,title,hist,xlabel,ylabel
        from numpy import array,exp,sum,arange,trapz,std,vstack,amin,amax,mean,unwrap,angle,where,median,sqrt,argmin

        figure(figsize=2*array([9.5,3]))

        t = this.gwfo.t
        beta_arr = this.__a_arr__
        beta_median = this.__a__
        cw = this.__cw__
        frmse_arr = this.__frmse_arr__
        maxdex = where( t>(t[0]+this.max_t_shift) )[0][0]
        index_range = range(maxdex+1) # NOTE that this line is prone to bugs for short data sets

        ax1 = subplot(131);clr = this.__clr__
        for k in range(len(this.__cw__)):
            plot( t[index_range], abs(beta_arr[k,:])-0*abs(this.__a__[k]), '-o',color=0.9*clr[k], label=this.qnminfo.icwmap[cw[k]], mfc='none', mec=0.9*clr[k] )
            axhline(abs(beta_median[k]),color=clr[k])
        xlim([ min( t[index_range]), max( t[index_range]) ])
        ylabel(r'$|A_{k}|$')
        legend(frameon=not True);title('amplitude')
        xlabel('$t/M$')
        # gca().set_yscale('log')

        ax2 = subplot(132)
        for k in range(len(cw)):
            plot( t[index_range], unwrap(angle(beta_arr[k,:]))-0*angle(this.__a__[k]), '-o', label=this.qnminfo.icwmap[cw[k]] ,color=0.9*clr[k], mfc='none', mec=0.9*clr[k])
            axhline( angle(beta_median[k]) ,color=clr[k])
        xlim([ min( t[index_range]), max( t[index_range]) ])
        title('Phase')
        ylabel(r'$\mathrm{arg}(A_{k})$')
        xlabel('$t/M$')

        ax3 = subplot(133)
        plot( t[index_range], frmse_arr, '-o', label=this.qnminfo.icwmap[cw[k]] ,color='k', mfc='none', alpha=0.9)
        axhline( median(frmse_arr) ,color='k')
        xlim([ min( t[index_range]), max( t[index_range]) ])
        title('frmse=%f'%this.frmse)
        ylabel(r'$\epsilon(t)$')
        xlabel('$t/M$')

        return ax1,ax2,ax3
    #
    def feval( this, t, gwfout=False, a=None, cw=None ):

        # Import useful things
        from numpy import sum,array,exp
        from nrutils import gwf

        # Handle optional inputs for QNM amplitudes, a, and frequencies, cw
        a = this.__a__ if a is None else a
        cw = this.__cw__ if cw is None else cw

        # Evaluate the model
        y = sum(  array([ a[k]*exp(t*1j*cw[k].conj()) for k in range(len(cw)) ])  , axis=0 )

        if not gwfout:
            ans = y
        else:
            wfarr = array( [t,y.real,y.imag] ).T
            from nrutils import gwf
            ans = gwf( wfarr, kind=this.gwfo.kind )

        #
        return ans

    #
    def plot( this, imrgwfo=None ):

        #
        from numpy import zeros,array
        from matplotlib.pyplot import figure,show,subplots

        # Setup plotting backend
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 18

        #
        ax = zeros( (6,) )

        #
        fig,axarr = subplots( 2,3, figsize=4*array([5,3]) )

        #
        this.__plotTDcomparison__( axarr[0,0] )
        this.__plotTDresiduals__( axarr[0,1] )
        this.__plotFitAmplitudes__( axarr[0,2] )
        #
        this.__plotCWContent__( axarr[1,0] )
        this.__plotTDAmpContent__( axarr[1,1] )
        this.__plotFDAmpContent__( axarr[1,2], imrgwfo=imrgwfo )

    #
    def __plotFDAmpContent__( this, ax=None, imrgwfo=None ):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,text
        from matplotlib.pyplot import ylim,axis,gca,xlim
        from kerr import rgb,pylim,lim
        from numpy import array,diff,sign,sin,cos,pi,linspace,ones,exp,ndarray,amin,amax,inf
        from nrutils import gwf

        #
        if ax is None:
            figure(figsize=[5,3])
            ax = gca()
        else:
            sca( ax )

        #
        y = this.gwfo.pad(new_length=5*len(this.gwfo.t),where='right')
        g = this.feval( y.t, gwfout=True )

        #
        clr = this.__clr__
        lw1, lw2 = 6, 1
        grey = 0.8 * array([1,1,1])

        #
        if imrgwfo.__class__.__name__ == 'gwf':
            plot( sign(this.homez[1])*imrgwfo.w, imrgwfo.fd_amp, ':', color='k', linewidth=1, label='IMR' )

        #
        plot( sign(this.homez[1])*y.w, y.fd_amp, '--',   color=grey, linewidth=lw1, label='NR' )
        plot( sign(this.homez[1])*g.w, g.fd_amp,   color='k', label='Fit' )
        min_amp,max_amp = inf,-inf
        for k,a in enumerate(this.__a__):
            cw = this.qnminfo.cwmap[this.amap[a]].conj()
            g_amp = abs( a/( cw-g.w ) )
            plot( sign(this.homez[1])*g.w, g_amp , color=clr[k] )
            min_amp = min( [min_amp,amin(g_amp)] )
            max_amp = max( [max_amp,amax(g_amp)] )

        # ax.set_yscale("log", nonposy='clip')

        #

        # pylim( g.w, g.fd_amp )
        ax.set_xlim( [0.05, sign(this.homez[1])*this.qnminfo.cwmap[this.homez].real + 0.8] )
        ax.set_ylim( [ min_amp, max_amp*1.5 ] )
        ax.set_yscale("log", nonposy='clip')
        ax.set_xscale("log", nonposy='clip')
        # pylim( this.qnminfo.cwmap[this.homez].real + array([-0.5,0.5]) , g.fd_amp )
        xlabel(r'$M\omega$' if sign(this.homez[1]) > 0 else r'$-M\omega$' )
        ylabel( '$|$'+y.kind+'$|$' )
        legend( frameon=False, loc=3 )


    #
    def __plotTDAmpContent__( this, ax=None ):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,text
        from matplotlib.pyplot import ylim,axis,gca
        from kerr import rgb,pylim,lim
        from numpy import array,diff,sign,sin,cos,pi,linspace,ones,exp,amin,amax

        #
        if ax is None:
            figure(figsize=[5,3])
            ax = gca()
        else:
            sca( ax )

        #
        y = this.gwfo
        g = this.feval( y.t, gwfout=True )

        #
        clr = this.__clr__
        lw1, lw2 = 6, 1
        grey = 0.8 * array([1,1,1])

        #
        plot( y.t, y.amp, '--',   color=grey, linewidth=lw1, label='NR' )
        plot( g.t, g.amp,   color='k', label='Fit' )
        for k,a in enumerate(this.__a__):
            plot( g.t, abs( a*exp(1j*g.t*this.qnminfo.cwmap[this.amap[a]].conj() ) ), color=clr[k] )

        ax.set_yscale("log", nonposy='clip')
        dy = (amax(y.amp)-amin(y.amp))*0.05
        ax.set_ylim([ amin(y.amp),amax(y.amp)+dy ])
        ax.set_xlim(lim(y.t))

        #
        #pylim( y.t, y.amp )
        xlabel(r'$t/M$')
        ylabel( '$|$'+y.kind+'$|$' )
        legend( frameon=False )


    #
    def __plotCWContent__(this,ax=None):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,text
        from matplotlib.pyplot import ylim,axis
        from kerr import rgb,pylim,lim
        from numpy import array,diff,sign,sin,cos,pi,linspace,ones

        #
        if ax is None:
            figure(figsize=[5,3])
        else:
            sca( ax )


        #
        homez = ( this.ll, this.mm, 0, int(sign(this.xf)) )
        A0 = this.iamap[ homez ]
        grey = 0.85*ones((3,))
        clr = this.__clr__

        #
        cw = this.qnminfo.cwmap[ this.homez ]
        plot( cw.real, -cw.imag, ('o' if this.homez[3]==1 else 's'), ms=28, mfc='none', mec='k', mew=3, alpha=0.125 )

        #
        for z in this.z:
            cw = this.qnminfo.cwmap[z]
            if (len(z)==4) and (z[1]==this.mm):
                l,m,n,p = z
                mkr = 'o' if p==1 else 's'
                mks = 20
                mkclr = 0.85*ones((3,)) if p==1 else 0.95*ones((3,))
            elif (len(z)==4) and (z[1]!=this.mm):
                l,m,n,p = z
                mkr = 'p' if p==1 else 'h'
                mks = 21
                mkclr = 0.65*ones((3,)) if p==1 else 0.75*ones((3,))
            else:
                mkr = '*'
                mkclr = ones((3,))
                mks = 22
                l,m,n,p,l2,m2,n2,p2 = z

            plot( cw.real, -cw.imag, mkr, ms=mks, mfc=mkclr, mec=0.5*mkclr, alpha=0.9 )

            tclr = 'k'
            if cw in this.__cw__:
                k = list(this.__cw__).index(cw)
                plot( cw.real, -cw.imag, 'o', ms=14, mec=this.__clr__[k]*0.9, mfc=clr[k], alpha=1 )
                tclr = 'w'

            text( cw.real, -cw.imag, '%i'%l, ha='center', va='center', alpha=1, clip_on=True, size=12, color=tclr, fontweight='normal' )

        #
        xlabel(r'$M\omega$')
        ylabel(r'$M/\tau$')

    #
    def __plotFitAmplitudes__(this,ax=None):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,text
        from matplotlib.pyplot import ylim,axis,gca
        from kerr import rgb,pylim,lim
        from numpy import array,diff,sign,sin,cos,pi,linspace

        #
        if ax is None:
            figure(figsize=[5,3])
            ax = gca()
        else:
            sca( ax )
            axis('equal')
            ax.yaxis.set_label_position("right")

        #
        th = 2*pi*linspace(0,1,256)
        x,y = cos(th),sin(th)
        plot(0,0,'+k',ms=10)
        plot( x,y, '-k', alpha=0.32, linewidth=1 )
        pylim( x,y,symmetric=True )

        #
        A0 = this.iamap[ this.homez ]
        clr = this.__clr__

        #
        for k,A in enumerate(this.__a__):
            a = A/abs(A0)
            b = a/abs(a)

            dl = 0.08; ha = 'left' if sign(a.real)<0 else 'right'

            if A==A0:
                plot( a.real, a.imag, 'ok', mfc='none', mec='k', ms=20, mew=3, alpha=0.125 )
            else:
                plot( b.real, b.imag, 'ok', mfc=clr[k], mec=0.6*clr[k], ms=4, alpha=0.8 )
            if abs(a)<1:
                plot( [a.real,b.real], [a.imag,b.imag], '--', alpha=0.2, color=0.6*clr[k] )

            if len( this.amap[A] ) == 4:
                # 1st Order Mode
                mkr = 'o'
                txt = '(%i,%i,%i,%i)'%this.amap[A]
            elif len( this.amap[A] ) == 8:
                # 2nd Order Mode
                mkr = 'h'
                txt = '(%i,%i,%i,%i)(%i,%i,%i,%i)'%this.amap[A]
            else:
                # Error
                error('Incorrectly formated mode coordinates: %s'%list(this.amap[A]))

            plot( [0,a.real], [0,a.imag], alpha=0.2, color=0.6*clr[k] )
            plot( a.real, a.imag, 'o', mfc = clr[k], mec=0.6*clr[k], ms = 8 )
            text( a.real-dl*sign(a.real), a.imag-dl, txt, alpha=0.6, clip_on=True, ha=ha )

        #
        title('Relative Amplitude Components')
        xlabel( r'Re $A_k/|A_{%i%i%i%i}|$'%this.amap[A0] )
        ylabel( r'Im $A_k/|A_{%i%i%i%i}|$'%this.amap[A0] )


    #
    def __plotTDresiduals__(this,ax=None):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,text
        from matplotlib.pyplot import ylim
        from kerr import rgb,pylim,lim
        from numpy import array,diff

        #
        y = this.gwfo
        g = this.feval( y.t, gwfout=True )

        #
        if ax is None:
            figure(figsize=[5,3])
        else:
            sca( ax )

        #
        clr = rgb(3)
        lw1, lw2 = 6, 1
        grey = 0.9 * array([1,1,1])
        alp = 1.0

        #
        s = ( g.y - y.y ) / y.amp
        sr = s.real
        sc = s.imag
        sa = abs(s)

        #
        plot( g.t, sa,   color=0.8*clr[1], linewidth=lw2, alpha=alp )
        plot( g.t,-sa,   color=0.8*clr[1], linewidth=lw2, label=r'$|\rho|$', alpha=alp )
        plot( g.t, sr,  color=clr[2], linewidth=lw2, label=r'Re($\rho$)', alpha=alp )
        plot( g.t, sc, color=clr[0], linewidth=lw2, label=r'Im($\rho$)', alpha=alp )

        #
        pylim( g.t, sa, symmetric=True )
        xlabel(r'$t/M$')
        text( g.t[0]+diff(lim(g.t))/2, min(ylim())+0.1*diff(ylim()), r'$\rho=$(Fit-Data)/|Data|', ha='center' )
        # ylabel( 'Fit - Data' )
        title( r'Fractional Residuals, $\langle frmse \rangle = %1.4f$'%(this.frmse) )
        legend( frameon=False, ncol=3 )

    #
    def __plotTDcomparison__(this,ax=None):

        #
        from matplotlib.pyplot import plot,sca,xlabel,ylabel,title,legend,gca
        from kerr import rgb,pylim
        from numpy import array

        #
        y = this.gwfo
        g = this.feval( y.t, gwfout=True )

        #
        if ax is None:
            figure(figsize=[5,3])
        else:
            sca( ax )

        #
        clr = rgb(3)
        lw1, lw2 = 6, 1
        grey = 0.9 * array([1,1,1])

        #
        plot( y.t, y.amp,   color=grey, linewidth=lw1, label='NR' )
        plot( y.t,-y.amp,   color=grey, linewidth=lw1 )
        plot( y.t, y.plus,  color=grey, linewidth=lw1 )
        plot( y.t, y.cross, color=grey, linewidth=lw1 )

        #
        plot( g.t, g.amp,   color=0.8*clr[1], linewidth=lw2 )
        plot( g.t,-g.amp,   color=0.8*clr[1], linewidth=lw2 )
        plot( g.t, g.plus,  color=clr[2], linewidth=lw2, label=r'Fit $+$' )
        plot( g.t, g.cross, color=clr[0], linewidth=lw2, label=r'Fit $\times$' )

        #
        pylim( g.t, g.amp, symmetric=True )
        xlabel(r'$t/M$')
        ylabel( y.kind )
        title( y.label )
        legend( frameon=False )

# ############################################################ %
''' Workflow class for applying fitting over NR cases (scentry objects) '''
# ############################################################ %
class modelrd:
    ''' Workflow class for applying fitting over NR cases (scentry objects).'''

    def __init__(this,                  # The current object
                 scentry_iterable=None, # A list of scentry (simulation catalog enrty) objects
                 T0=None,                 # The starting time relative to the peak luminosity
                 T1=None,               # Ending time of RD fitting region
                 workdir=None,          # Highest level directory for file IO
                 show=False,            # Toggle for showing plots
                 noplots=False,         # Toggle for creating and saving plots
                 lmlist=None,           # List of spherical multipoles to consider
                 clean=False,           # Toggle for removing intermediate files
                 keyword=None,          # Label for output files
                 greedy=True,           # Toggle for use of greedy fitting
                 use_peak_strain=True,  # Toggle for listing time relative to peak strain
                 scri=False,            # Toggle to extrapolate fitting results to infinity
                 verbose=True):         # Let the people know


        # Let the people know (what was input)
        if verbose:
            print '\n\n%s\n## \t %s \n%s\n\n' % ( 'wwwWv~-'*6, yellow('MODEL RINGDOWN'), 'wwwWv~-'*6 )
            print 'Settings\n%s'%('--'*20)
            for k in dir():
                if (eval(k) is not None) and (eval(k) is not False) and not ('this' in k):
                    print '## %s = %s' % ( cyan(str(k)), yellow(str(eval(k))) )
            print '\n\n'

        # Import unseful things
        import pickle
        from nrutils import gwylm,scsearch
        from os.path import expanduser,isfile,isdir
        from os import remove as rm
        from shutil import rmtree as rmdir
        from numpy import array


        # Define domain vairables that can be understood by the latex
        # code, as well as the make_domain code. NOTE that the point of this line
        # is to centralize the definition of which variables will be used for modeling
        this.model_domain_variables = [ 'eta','chi_s' ]

        # NOTE that model quality (particularly overmodeling) is affected by this value
        this.fitatol = 1e-2 # 1e-2
        # Location of fitting region in M relative to peak strain
        this.T0 = 20 if T0 is None else T0
        this.T1 = T1

        # Toggle to use p_range determined by jf OR the general (-1,1)
        #   * If true, prange is the sign of jf, else it is (-1,1)
        #   * This option is applied externally to the qnmfit class via the prange input
        # NOTE: setting use_spin_prange to False has been found to reeeally muck things up
        this.use_spin_prange = True

        # Handle the prange property using the use_spin_prange value
        if this.use_spin_prange:
            # Setting prange None here means that the spign of
            # the final spin will be used in the qnmfit class
            this.prange = None
        else:
            # Both counter and co-rotating QNM will be used
            this.prange = [-1,1]

        # Default size for figures
        this.figsize = 2*array([4,2.8])
        # Time step to use for all data
        this.dt = 0.35
        # Tag for whether or not to use greedy fitting in the qnmfit process
        this.greedy = greedy
        # Store time listing option
        this.use_peak_strain = use_peak_strain
        # Toggle for removing junk radiation
        # NOTE that removing junk radiation affecs the peak strain location and thus the values of teh QNM amplitudes. Counterintuitively, cleaning appear to (slightly) negatively affect results.
        this.use_cleaned_waveforms = False

        # Double check that cleaning of intermediate data really is desired
        if clean:
            clean = 'y' in raw_input(warning('The %s option has been detected. Do you really wish to remove internediate data files? [%s] %s  '%( red(bold('CLEAN')), bold(yellow('Yes')+'/'+yellow('no')) ,cyan('If yes, note that it may take a long time to regenerate them using this tool.')),output_string=True) ).lower()

        # NOTE that if a new scentry_iterable is given, then previous working data will always be cleaned
        if not (scentry_iterable is None):
            alert( '%sscentry_iterable is given, then previous working data will always be cleaned'%(red('Note: ')) )
            clean = True

        # Internalize clean and verbose inputs
        this.verbose,this.clean,this.keyword,this.scri = verbose,clean,keyword,scri

        # Handle the input keyword option and format it to be joined with file names
        keyword = '' if keyword is None else keyword+'_'

        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        #[A]# Setup directories for storing intermediate data                                        #
        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        workdir = '~/KOALA/kerr_dev/workflows/modelrd/' if (workdir is None) else workdir
        workdir = expanduser( workdir ).replace('//','/')
        mkdir(workdir,verbose=True)
        this.workdir = workdir
        # Make directory to store tempoary data
        bindir = this.workdir + '/bin/'
        bindir.replace('//','/')

        # If clean, remove all intermediate data files
        if this.clean and isdir(bindir):
            import glob, os
            map(os.remove, glob.glob("%s/*%s.bin"%(bindir,keyword)))

        # If the bindir does not exist, make it, and store it to the current object
        mkdir(bindir,verbose=True)
        this.bindir = bindir

        # Determine if qnmfit data exists; clean if desired
        this.qnmfit_data_path = this.bindir + this.keyword + 'qnmfit.bin'
        if clean and isfile( this.qnmfit_data_path ):
            rm( this.qnmfit_data_path )

        # Handle list of spherical modes to use
        if lmlist is None:
            # lmlist_,lmlist = [ (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (5,4) ],[]
            # lmlist_,lmlist = [ (2,2), (2,1) ],[]
            lmlist_,lmlist = [ (2,2), (2,1), (3,2), (4,3), (3,3), (4,4), (5,5) ],[]
            # Add the m<0 counterparts for consistency checking
            if not ( (2,1) in lmlist_ ): lmlist_ += [(2,1)]
            for lm in lmlist_:
                lmlist.append( (lm[0],lm[1]) )
                # lmlist.append( (lm[0],-lm[1]) )
            # Sort and set
            lmlist = sorted( list( set(lmlist) ) )

        # NOTE that we store the list of spherical eigenvalues here to be used later
        this.lmlist = lmlist

        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        #[B]# Data Collection will proceed in a nested rather than modular fashion to conserve disk space
        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        pad = '####'*12+'#'
        if this.verbose: print '%s\n# Processing Fit Data from Simulations\t\t#\n%s'%(pad,pad)

        # If the core data file does not extist, cull QNM fit data anew
        if not isfile( this.qnmfit_data_path ):

            #
            alert('No scentry_iterable input found. A default list of simulations will now be collected.')
            if scentry_iterable is None:
                from nrutils import scsearch,jf14067295,Mf14067295
                from numpy import isnan,array
                # Get all gt simulations of interest
                scentry_iterable = scsearch(keyword=('hr','hrq','sq'),nonprecessing=True,verbose=False,unique=True,institute='gt')
                scentry_iterable = scsearch(catalog = scentry_iterable, keyword=('bradwr'),nonprecessing=True,verbose=True)
                # Add the BAM Runs
                bam_runs = scsearch(keyword='silures',nonprecessing=True,verbose=True,unique=True)
                # Concat
                scentry_iterable = bam_runs + scentry_iterable
                # NOTE that the remnant properties from the BAM runs cannot be trusted, so we will use a final mas and spinf fit here
                for e in scentry_iterable:
                    e.mf,e.xf = Mf14067295(e.m1,e.m2,e.X1[-1],e.X2[-1]),jf14067295(e.m1,e.m2,e.X1[-1],e.X2[-1])
                    e.Sf = e.mf*e.mf*array([0,0,e.xf])
                #
                scentry_iterable = this.filter_scentry_resolution(scentry_iterable,res_min=160)

            # NOTE that the values in scentry_iterable will be stored in a dicionary call qnmfit_by_simulation
            this.cull_qnmfit_by_simulation(scentry_iterable)
        else:
            # Try to load pre-calculated qnmfit objects
            this.load_qnmfit_by_simulation()
            # NOTE that at this point, this.qnmfit_by_simulation should be set to the contents of this.qnmfit_data_path


        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        #[C]# Organize the data into lists that make modeling easier                                 #
        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        if this.verbose: print '%s\n# Organizing Fit Data for Modeling\t\t#\n%s'%(pad,pad)
        this.organize()

        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        #[D]# Model QNM amplitudes over a chosen domain                                              #
        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        if this.verbose: print '%s\n# Modeling QNM Complex Amplitudes Over a Chosen Domain\t\t#\n%s'%(pad,pad)
        this.qnm_manifold_learn()

        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        #[E]# Document the fit results: latex, python                                                #
        #%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%###%%#
        # this.document()

    # Given list of scentry objects, load and fit QNMs, then store(write) using pickle.
    def cull_qnmfit_by_simulation(this,scentry_iterable):
        '''Given list of scentry objects, load and fit QNMs, then store(write) using pickle.'''

        # Import useful things
        import pickle

        # QNM fit information will be stored to the curernt object in this dictionary
        this.qnmfit_by_simulation = {}
        this.qnmfit_at_infinity_by_simulation = {}

        # For all scentry objects
        alert('Collecting qnmfit data ...')
        n = len( scentry_iterable )
        for k,e in enumerate(scentry_iterable):

            # Let the people know
            simname = e.raw_metadata.source_dir[-1].split('/')[-1] if e.raw_metadata.source_dir[-1][-1]!='/' else e.raw_metadata.source_dir[-1].split('/')[-2]
            if this.verbose: print '%s\n# Processing %i/%i: %s (%s)\n%s'%('===='*12,k+1,n,cyan(e.label),green(simname),'===='*12)

            # Process the current scentry object, but don't save. We will save the list when all scetry objects have been processed.
            if not this.scri:
                this.process_scentry( e,load_and_save=False )
            else:
                this.process_scentry_at_infinity( e,load_and_save=False )

        # Pickle the qnmfit list
        this.save_qnmfit_by_simulation()

    # Try to save the qnmfit_by_simulation list to this.qnmfit_data_path
    def save_qnmfit_by_simulation(this):
        # Import something useful
        import pickle
        '''Try to save the qnmfit_by_simulation list to this.qnmfit_data_path'''
        # Pickle the qnmfit list
        if this.verbose: print '%s\n# Saving fit information to file: "%s"\n%s'%('####'*12,cyan(this.qnmfit_data_path),'####'*12)
        with open(this.qnmfit_data_path, 'wb') as fit_file:
            pickle.dump( (this.qnmfit_by_simulation,this.qnmfit_at_infinity_by_simulation) , fit_file, pickle.HIGHEST_PROTOCOL )

    # Try to load pre-calculated qnmfit objects
    def load_qnmfit_by_simulation(this):
        # Impor useful things
        import pickle
        # Try to load pre-calculated qnmfit objects
        if this.verbose: alert('Loading pre-calculated qnmfit data from: "%s"'%yellow(this.qnmfit_data_path))
        with open( this.qnmfit_data_path , 'rb') as fit_file:
            this.qnmfit_by_simulation,this.qnmfit_at_infinity_by_simulation = pickle.load( fit_file )
        if this.verbose: alert('Loading complete: %s QNM Fit instances loaded.'%(green(str(len(this.qnmfit_by_simulation)))))

    # Fit a Simulations dataset after the core data collection process has already finished and store it to the core datafile
    def process_scentry( this,  # The current object
                         e_,    # The scentry class object to be fit
                         greedy=None, # Toggle for use of greedy algorithm in qnmfit; default is True
                         load_and_save=True): # Toggle for loading and saving of qnmfit_by_simulation data.

        # Import useful things
        from nrutils import gwylm

        # Try to load pre-calculated qnmfit objects
        if load_and_save: this.load_qnmfit_by_simulation()
        # NOTE that at this point, this.qnmfit_by_simulation should be set to the contents of this.qnmfit_data_path

        # Handle greedy input
        greedy = this.greedy if greedy is None else greedy

        # #
        # simname = e.raw_metadata.source_dir[-1].split('/')[-1] if e.raw_metadata.source_dir[-1][-1]!='/' else e.raw_metadata.source_dir[-1].split('/')[-2]
        # if this.verbose: print '%s\n# Processing %i/%i: %s (%s)\n%s'%('===='*12,k+1,n,cyan(e.label),green(simname),'===='*12)

        # Load relevant waveforms ...

        # -------------------------------------------------------- #
        if this.verbose: print '>> Loading NR data ...',
        # NOTE that automatic strain calculation is disabled becuase ringdwown fitting will be applied to Psi4; also note that strain conversion is easy for exponentials
        # NOTE that we do wish to clean the waveforms as junk radiation afftects the location of peak strain
        a = gwylm( e_, lm=this.lmlist, dt=this.dt, clean = this.use_cleaned_waveforms )
        # -------------------------------------------------------- #

        if this.verbose: print green('Done.')
        a.calcflm()

        # -------------------------------------------------------- #
        # Select the Ringdown portion
        if this.verbose: print '>> Selecting Ringdown ...',
        b = a.ringdown(T0=this.T0,use_peak_strain=this.use_peak_strain,verbose=this.verbose)
        if this.verbose: print green('Done.')
        # -------------------------------------------------------- #

        # Try to determine of the scentry a;ready exists within the object's dictionary.
        # This is not as trivial as referencing the dictionary with the input scentry as
        # the class objects were made at different times and thus represent different
        # memory locations.
        e_ref = e_
        for e_tmp in this.qnmfit_by_simulation:
            if e_tmp.raw_metadata.source_dir == e_.raw_metadata.source_dir:
                e_ref = e_tmp

        # Initiate this simulation's list of qnmfit objects
        this.qnmfit_by_simulation[e_ref] = []

        # For all psi4 multipoles, fit qnms
        if this.verbose: print '>> Fitting Spherical Multipoles:'
        for y in b.ylm:

            # Let the people know
            if this.verbose: print '\n'+cyan('****'*10)+'\n>> Processing Spherical Multipole: '+magenta('(%i,%i)'%(y.l,y.m))+'\n'+cyan('****'*10)

            # -------------------------------------------------------- #
            # Create a qnm fit for this multipole
            f = qnmfit(y,verbose=this.verbose,greedy=greedy,prange=this.prange)
            # -------------------------------------------------------- #

            # Store to the current object
            this.qnmfit_by_simulation[e_ref].append( f )

        # Save the current object's qnmfit_by_simulation list
        if load_and_save: this.save_qnmfit_by_simulation()

        #
        return None

    # Fit a Simulations dataset after the core data collection process has already finished and store it to the core datafile
    def process_scentry_at_infinity( this,  # The current object
                         e_,    # The scentry class object to be fit
                         greedy=None, # Toggle for use of greedy algorithm in qnmfit; default is True
                         load_and_save=True): # Toggle for loading and saving of qnmfit_by_simulation data.

        # Import useful things
        from nrutils import gwylm
        from copy import deepcopy as copy
        from numpy import array
        import dill, pickle
        from matplotlib.pyplot import show,savefig,gcf,close

        # Try to load pre-calculated qnmfit objects
        if load_and_save: this.load_qnmfit_by_simulation()
        # NOTE that at this point, this.qnmfit_by_simulation should be set to the contents of this.qnmfit_data_path

        # Handle greedy input
        greedy = this.greedy if greedy is None else greedy

        # Get extraction radii map
        extraction_pars = array(sorted(e_.extraction_radius_map.keys()))
        extraction_rads = array([ e_.extraction_radius_map[k] for k in extraction_pars ])

        # NOTE that we will use prior knowledge about which extraction radii are "good" to mask the list of possible opeitons
        mask = (extraction_rads>40) & (extraction_rads<100)
        extraction_pars = extraction_pars[mask]
        extraction_rads = extraction_rads[mask]

        #
        print '>> The following extraction radii will be processed:%s' % extraction_rads

        #
        output_intermediate_files = False

        # Load relevant waveforms ...

        # Try to determine of the scentry already exists within the object's dictionary.
        # This is not as trivial as referencing the dictionary with the input scentry as
        # the class objects were made at different times and thus represent different
        # memory locations.
        e_ref = e_
        if load_and_save:
            for e_tmp in this.qnmfit_at_infinity_by_simulation:
                if e_tmp.raw_metadata.source_dir == e_.raw_metadata.source_dir:
                    e_ref = e_tmp

        # Define output dir for scri diagnostic files
        if output_intermediate_files:
            outdir = this.bindir + '/%sscri/'%this.keyword + '/%s/'%e_.simname
            outdir = outdir.replace('//','/')
            mkdir(outdir,verbose=True)

        # Initiate this simulation's list of qnmfit objects
        this.qnmfit_by_simulation[e_ref] = []
        this.qnmfit_at_infinity_by_simulation[e_ref] = []

        # For each extraction radius, fit the ringdown for an ll,mm multipole
        qnmfit_by_extraction_radius = {}
        ref_qnmfit_by_mode = {}
        ref_qnmfit = None
        qnmfit_by_lm = { lm:{} for lm in this.lmlist }
        for k,r in enumerate(extraction_rads):
            print magenta('%s\n## Loading and fitting for r=%1.2f\n%s' % ('####'*10,extraction_rads[k],'####'*10))
            g = gwylm( e_, extraction_parameter=extraction_pars[k], lm=this.lmlist, dt=this.dt, clean = this.use_cleaned_waveforms )
            b = g.ringdown(T0=this.T0,use_peak_strain=this.use_peak_strain)
            #
            qnmfit_by_extraction_radius[r] = {}
            for y in b.ylm:
                # -------------------------------------------------------- #
                # Create a qnm fit for this multipole
                f = qnmfit(y,verbose=False,greedy=greedy,prange=this.prange)
                # -------------------------------------------------------- #
                # Extract the spherical coordinates
                lm = (y.l,y.m)
                # Store to the current fit object for the current extraction radius
                qnmfit_by_lm[lm][r] = f
                # If r is the reference value, then store this fit to the global holder
                # NOTE that the values stored in the global holder will be modified to contain the extrapolated answer
                if r==75:
                    # this.qnmfit_at_infinity_by_simulation[e_ref].append( copy(f) )
                    # this.qnmfit_by_simulation[e_ref].append(f)
                    ref_qnmfit_by_mode[lm] = f

        # Organize fitting results

        # For all spherical modes
        for lm in qnmfit_by_lm:

            print magenta('%s\n## Collecting and Extrapolating for [l,m]=[%i,%i]\n%s' % ('####'*10,lm[0],lm[1],'####'*10))

            # Find all QNM indeces of relevance
            big_zlist = []
            for r in qnmfit_by_lm[lm]:
                big_zlist += qnmfit_by_lm[lm][r].iamap.keys()
            big_zlist = list(set(big_zlist))

            # Find QNMs found at all radii
            z_list = []
            for z in big_zlist:
                state = True
                for r in qnmfit_by_lm[lm]:
                    state = state and ( z in qnmfit_by_lm[lm][r].iamap.keys() )
                if state: z_list.append( z )


            # Extract the estimate at scri and store


            # NOTE that there is ONE fit object for EACH lm value
            f = ref_qnmfit_by_mode[lm]
            g = copy(f)
            #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
            this.qnmfit_at_infinity_by_simulation[e_ref].append( f )
            # NOTE that the amplitude values of f are changed below for all z
            this.qnmfit_by_simulation[e_ref].append( g )
            #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

            # Extrapolate the QNMs for this lm to infinity
            for z in z_list:

                # Get QNM amplitude for all radii
                ak = []
                for r in extraction_rads:
                    ak.append( qnmfit_by_lm[lm][r].iamap[z] )

                # Given extraction raddi, r, and ampls, ak, extrapolate
                domain = 1.0 / array(extraction_rads)
                scalar_range = array(ak)

                # Find the most likely outlier in the scalar_range by minimizing the set's standard deviation
                _,mask = single_outsider( abs(scalar_range) )
                domain = domain[mask]
                scalar_range = scalar_range[mask]

                # scri = gmvpfit( domain, scalar_range, maxdeg=2, fitatol=1e-2, initial_boundary=['K'], apply_negative=False, verbose=True )

                # Force desired behavior at infinity
                scri = mvpolyfit( domain, scalar_range, basis_symbols=['K','00'],verbose=True )

                # Set the amplitude value to the extrapolated one
                f.iamap[z] = scri.coeffs[ list(scri.basis_symbols).index('K') ]
                alert('[%s] The r=75 value is %s, and the r=Inf value is %s'%(list(z),red(complex2str(f.iamap[z])),magenta(complex2str(g.iamap[z]))) )

                # Save plot of extrapolation
                if output_intermediate_files:
                    scri.plot(show=False,fit_xmin=0)
                    image_path = outdir + 'plot_%s_ll%imm%il%im%in%ip%i.pdf' % ( e_.simname, lm[0],lm[1],z[0],z[1],z[2],z[3])
                    savefig( image_path, bbox_inches='tight', pad_inches=0 )
                    # NOTE, close them there figs! (This momement of American dialect was brought to you by "Lionel London needs some sleep." If you see Lionel, give him a pillow.)
                    close('all')

                # Save fit data
                if output_intermediate_files:
                    odf =  outdir + 'gmvpfit_%s_ll%imm%il%im%in%ip%i.bin' % ( e_.simname, lm[0], lm[1], z[0], z[1], z[2], z[3] )
                    with open( odf, 'wb') as fid:
                        pickle.dump( (e_,scri) ,fid )

        # Save the current object's qnmfit_by_simulation list
        if load_and_save: this.save_qnmfit_by_simulation()

        #
        return None

    # Organize the fit data and store in the current object
    def organize(this):
        '''Organize the fit data and store in the current object'''
        #
        import pickle
        from numpy import angle,exp,pi,linspace,array,ceil,sign,floor
        from os.path import expanduser,isfile
        from os import remove as rm

        # Determine if stored core data exists; clean if desired
        core_data_path = this.bindir + this.keyword + 'core.bin'
        if this.clean and isfile( core_data_path ):
            rm( core_data_path )

        if isfile( core_data_path ):
            # Load it
            alert('Loading pre-tabulated core data from: "%s"'%yellow(core_data_path))
            with open( core_data_path , 'rb') as core_file:
                this.data = pickle.load( core_file ) # NOTE that this line must be consitent with the conditional case below in its assignment of "this.data"
            alert('Loading of core data complete. We may now Proceed to modeling.')
        else:
            # Generate it
            alert('No pre-tabulated core data found. We will now generate tabulated core data and store it to "%s"'%yellow(core_data_path))
            # Initialize bulk data holders
            per_qnm = {}
            per_sim = { 'chi1':[],'chi2':[], 'm1':[],'m2':[], 'jf':[],'Mf':[],'eta':[],'sim_id':[] }
            # NOTE that even though per_sim and per_qnm will be referencced below, the data *will* be stored to this.data
            this.data = { 'per_sim':per_sim, 'per_qnm':per_qnm }
            #
            for k,e in enumerate(this.qnmfit_by_simulation):

                #
                alert('Tabulating data for case %s'%(yellow('%i/%i'%(k+1,len(this.qnmfit_by_simulation)))),'modelrd.organize')

                # Use the simulation directory as a unique sim ID
                sim_id = e.simdir()
                if this.verbose: print '## Working "%s"'%magenta(sim_id)

                #
                _m1,_m2 = e.m1,e.m2
                _S1,_S2 = e.S1,e.S2
                _chi1,_chi2 = _S1/(_m1**2), _S2/(_m2**2)

                # Initialize holder for this simulation
                # NOTE that data is only stored for a simulation the first time it is recognized
                if not ( sim_id in per_sim['sim_id'] ):
                    #
                    per_sim['m1'].append(_m1)
                    per_sim['m2'].append(_m2)
                    per_sim['eta'].append( _m1*_m2 / (_m1+_m2)**2 )
                    per_sim['chi1'].append( _chi1 )
                    per_sim['chi2'].append( _chi2 )
                    per_sim['jf'].append(e.xf)
                    per_sim['Mf'].append(e.mf)
                    per_sim['sim_id'].append(sim_id)

                # Get (Estimate!) the orbital phase value to use for this simulation's ringdown at the start of the fitting region

                # Handle extrap to infinity flow
                if not this.scri:
                    qnmfit_by_simulation = this.qnmfit_by_simulation
                else:
                    qnmfit_by_simulation = this.qnmfit_at_infinity_by_simulation

                #
                f = [ k for k in qnmfit_by_simulation[e] if (k.ll,k.mm)==(2,1) ][0]
                p = int(sign(f.xf))
                a2101 = f.iamap[2,1,0,p] / (f.qnminfo.cwmap[2,1,0,p]**2)
                phi0_2101 = angle(a2101) # NOTE that there is a NO pi/n ambiguity here!!!
                #
                f = [ k for k in qnmfit_by_simulation[e] if (k.ll,k.mm)==(2,2) ][0]
                p = int(sign(f.xf))
                a2201 = f.iamap[2,2,0,p] / (f.qnminfo.cwmap[2,2,0,p]**2)
                phi0_2201 = angle(a2201)/2 # NOTE that there is an overall pi ambiguity here!!!

                # NOTE that here we estimate the value of phi0_2201's phase ambiguity:
                # * phi0_2201 \approx phi0 + pi*b, where b is an integer
                # * phi0_2101 \approx phi0
                # --> b \approx (phi0_2201-phi0_2101) / pi
                b = floor( (phi0_2201 - phi0_2101)/pi )
                # NOTE that the prior use of ceil below incurred a pi pase error for the odd m QNMs
                # b = ceil( (phi0_2201 - phi0_2101)/pi )
                phi0 = phi0_2201 - pi*b
                print '>> b = %i'%b

                # Precalculate the theta values that will be used to estimate inner products between spherical and spheroidal harmonics
                _theta = pi*linspace(0,1,2**11)

                # For all spherical modes uesd in this simulation
                for f in qnmfit_by_simulation[e]:
                    if this.verbose: print '**\t(ll,mm) = (%s,%2s) \t**'%(yellow(str(f.ll)),green(str(f.mm)))
                    # For all QNMs for this single Spherical Mode
                    for k in f.iamap:

                        # Sort by compound index: jk = ( (ll,mm),(l,m,n,p) )
                        j = (f.ll,f.mm)             # the spherical mode ll,mm
                        jk = ( j,k )                 # the spheroidal mode l,m,n,p

                        # Initialize the holder for this net QNM coordinate
                        if not ( jk in per_qnm ):
                            per_qnm[jk] = { 'chi1':[],'chi2':[], 'm1':[],'m2':[], 'jf':[],'Mf':[],\
                             'Ak':[], 'eta':[], 'cw':[], 'phi0':[], 'sim_id':[], 'homez':[], 'is_not_q1ns':[] }

                        # Calculate useful bits
                        cw = f.qnminfo.cwmap[k]
                        # Store the RAW fit amplitude (further manipulations such as scaling away leading order PN should be applied before modeling)
                        _Ak = f.iamap[k]

                        # ------------------------------------------------------ #
                        # Store Stuff
                        # ------------------------------------------------------ #
                        # Initial binary parameters
                        per_qnm[jk]['m1'].append(_m1)
                        per_qnm[jk]['m2'].append(_m2)
                        per_qnm[jk]['eta'].append( _m1*_m2 / (_m1+_m2)**2 )
                        per_qnm[jk]['chi1'].append( _chi1 )
                        per_qnm[jk]['chi2'].append( _chi2 )
                        per_qnm[jk]['jf'].append(f.xf)
                        per_qnm[jk]['Mf'].append(f.Mf)
                        # QNM Information
                        per_qnm[jk]['cw'].append(cw)         # QNM frequency
                        per_qnm[jk]['Ak'].append(_Ak)        # Raw fit amplitude from Psi4 multipole
                        per_qnm[jk]['phi0'].append(phi0)     # reference phase
                        # Simulation Identifiers
                        per_qnm[jk]['sim_id'].append(sim_id) # Simulation directory name (not path)
                        per_qnm[jk]['homez'].append(f.homez) # Spherical multipole coordinates for this QNM
                        per_qnm[jk]['is_not_q1ns'].append(\
                        (abs(_m1-_m2)+sum(abs(_chi1))+\
                        sum(abs(_chi2))) < 1e-3 )           # Needed to mask away values for the equal mass nonspinning case, where some QNM are not excited do to geometric symmetry

            # Convert lists to numpy arrays for future utility
            for k in per_sim:
                per_sim[k] = array( per_sim[k] )
            for jk in per_qnm:
                for h in per_qnm[jk]:
                    per_qnm[jk][h] = array( per_qnm[jk][h] )

            # Pickle the bulk data
            if this.verbose: print '%s\n# Saving core data information to file: "%s"\n%s'%('####'*12,cyan(core_data_path),'####'*12)
            with open(core_data_path, 'wb') as core_file:
                pickle.dump( this.data , core_file, pickle.HIGHEST_PROTOCOL )

    # Calculate the leading order pn scaling with the assumption that it applies to QNM due to geometry
    def get_leading_pn_scale(this,jk_):
        '''Given a modelrd object and an iterable of QNM indeces (l,m,n,p)
        return the expected leading order analytic scaling suggested by PN'''
        # Import useful things
        from numpy import sqrt
        # Unpack the modelrd object
        chi1 = this.list_qnm_data( jk_, 'chi1' )
        chi2 = this.list_qnm_data( jk_, 'chi2' )
        m1 = this.list_qnm_data( jk_, 'm1' )
        m2 = this.list_qnm_data( jk_, 'm2' )
        # NOTE that jk_ = [(ll,mm),(l,m,n,p)]
        # NOTE that the value of m not mm is used below as some QNMs may not have m==mm if they are kick-modes
        mm = jk_[1][1]
        # Distill initial binary parameters
        M = m1+m2
        dM = ( m1 + m2*(-1)**mm ) / M # NOTE that dM below is 1 if mm is even
        _dM= ( m1 - m2 ) / M #
        eta = m1*m2/(M*M)
        chi_s = (m1*chi1 + m2*chi2) / M
        chi_a = (m1*chi1 - m2*chi2) / M

        # NOTE that the spinc "reduced" parameters mentioned in eq 5.8 of arxiv:1107.1267 is used below
        chi = chi_s + _dM*chi_a - eta*chi_s*76.0/113.0

        # Calculate and return scaling suggested by PN
        # * NOTE that the  ( eta*dM + chi) has been added heuristically based on thinking about limits
        # ans = eta*(dM + chi_a)
        ans = this.pn_scale_fun( eta,jk_ )
        # ans = eta
        return ans

    # Calculate the mm dependent PN scaling
    # NOTE that this function exists abstractly to assist the creation of a range_map for gmvpfit
    def pn_scale_fun( this, eta, jk_ ):
        from numpy import sqrt
        mm = jk_[1][1]
        u = int( (1.0-(-1)**mm)/2.0 )
        dM = sqrt( 1.0 - 4.0*eta*u )
        return eta*dM

    # Calculate the QNM strain Amplitudes divided by the pn scale
    def get_pn_scaled_strain_amplitudes(this,jk_):
        '''
        Given jk_=( (ll,mm), (l,m,n) ), get all qnm Ak with p in [1,-1], and then
        calculate teh QNM strain Amplitudes divided by the pn scale.
        '''
        # NOTE that jk = ( (ll,mm), (l,m,n,p) )
        # Import useful things
        from numpy import exp,angle,pi
        # Retrieve the rew fit QNM amplitudes, they are Psi4 Amplitudes
        Ak_psi4 = this.list_qnm_data( jk_, 'Ak' )
        # Convert to strain by dividing by frequency squared
        cw = this.list_qnm_data( jk_, 'cw' )
        Ak_strain = Ak_psi4 / ( 1j*cw * 1j*cw )
        # Scale away expected leading PN behavior
        # NOTE: it is expected that doing so will result in a simpler range to model
        ak = Ak_strain / this.get_leading_pn_scale(jk_)
        # NOTE that the step below is a processing step:
        m,phi0 = jk_[0][1],this.list_qnm_data( jk_, 'phi0' )
        ak *= exp( -1j*m*phi0 )
        # NOTE that the values above may need to be masked to remove equal mass nonspinning cases, where QNMs with odd m are not excited
        ans = ak
        return ans

    # Get a range_map to be used with gmvpfit
    def get_range_map( this, jk_ ):
        '''Get a range_map to be used with gmvpfit'''
        # NOTE that here, dom[0] must be eta, the symmetric mass ratio
        forward =  lambda dom,ran:         ran         / this.pn_scale_fun(dom[:,0],jk_)
        backward = lambda dom,forward_ran: forward_ran * this.pn_scale_fun(dom[:,0],jk_)
        range_map = { 'forward':forward, 'backward':backward }
        # Return the answer
        ans = range_map
        return ans
    #
    def get_forward_range_map( this, jk_ ):
        return None

    # Calculate teh QNM strain Amplitudes divided by the pn scale
    def get_phase_aligned_strain_amplitudes(this,jk_):
        '''
        Given jk_=( (ll,mm), (l,m,n) ), get all qnm Ak with p in [1,-1], and then
        calculate teh QNM strain Amplitudes divided by the pn scale.
        '''
        # NOTE that jk = ( (ll,mm), (l,m,n,p) )
        # Import useful things
        from numpy import exp,angle,pi
        # Retrieve the rew fit QNM amplitudes, they are Psi4 Amplitudes
        Ak_psi4 = this.list_qnm_data( jk_, 'Ak' )
        # Convert to strain by dividing by frequency squared
        cw = this.list_qnm_data( jk_, 'cw' )
        Ak_strain = Ak_psi4 / ( 1j*cw * 1j*cw )
        # NOTE that the step below is a processing step:
        m,phi0 = jk_[0][1],this.list_qnm_data( jk_, 'phi0' )
        Ak_strain *= exp( -1j*m*phi0 )
        # NOTE that the values above may need to be masked to remove equal mass nonspinning cases, where QNMs with odd m are not excited
        ans = Ak_strain
        return ans

    # Given jk_=( (ll,mm), (l,m,n) ), return all qnm listed data with p in [1,-1]
    def list_qnm_data( this, jk_, name ):
        '''Given jk_=( (ll,mm), (l,m,n) ), return all qnm listed data with p in [1,-1]'''
        from numpy import hstack,array

        ll,mm = jk_[0];

        if len(jk_[1]) == 3:
            l,m,n = jk_[1]
            jk_p1 = ( (ll,mm), (l,m,n,1) )
            jk_p2 = ( (ll,mm), (l,m,n,-1) )
        else:
            l1,m1,n1,l2,m2,n2 = jk_[1]
            jk_p1 = ( (ll,mm), (l1,m1,n1, 1,l2,m2,n2, 1) )
            jk_p2 = ( (ll,mm), (l2,m2,n2,-1,l2,m2,n2,-1) )

        ans_p1 = array( this.data['per_qnm'][jk_p1][name] ) if jk_p1 in this.data['per_qnm'] else array([])
        ans_p2 = array( this.data['per_qnm'][jk_p2][name] ) if jk_p2 in this.data['per_qnm'] else array([])
        if 'chi' in name:
            ans_p1,ans_p2 = ans_p1[:,-1] if jk_p1 in this.data['per_qnm'] else array([]), ans_p2[:,-1] if jk_p2 in this.data['per_qnm'] else array([])
        ans = hstack( [ans_p1,ans_p2] )
        return ans

    # Learn the various manifolds associated with fits
    def qnm_manifold_learn( this,               # The current object
                            store = None,       # Toggle for stoing modeling results in various formats
                            fitatol = None,     # Tolerance parameter for modeling
                            verbose = None ):   # Toggle to let the people know
        '''
        For every parameter in this.data, apply manifold learning and output related information:
            * Plots of leaning space and practical space fits on data
            * Python modules for fits
            * latex files for fitting formula
        '''

        # Import useful things
        from numpy import vstack,isfinite,mod,sign
        import pickle
        from os.path import expanduser,isfile
        from os import remove as rm
        import sys

        # Determine if stored model data exists; clean if desired
        models_data_path = this.bindir + this.keyword + 'models.bin'
        if this.clean and isfile( models_data_path ):
            rm( models_data_path )

        # Handle the fitatol to be used
        this.fitatol = this.fitatol if fitatol is None else fitatol

        # IF the models datafile does not exist
        if isfile( models_data_path ):

            # Load it
            if this.verbose: alert('Loading pre-learned models from: "%s"'%yellow(models_data_path))
            with open( models_data_path , 'rb') as models_file:
                this.models = pickle.load( models_file ) # NOTE that this line must be consitent with the conditional case below in its assignment of "this.data"
            if this.verbose: alert('Loading of models complete. We may now Proceed to documenting.')

        else: # ELSE, make a model for each QNM amplitude, and store the realted data

            # Define a holder for QNM models
            this.models = {}

            # Prepare a list of all QNM coordinates:
            # * There are 6 in total: the spherical ll,mm and the spheroidal l,m,n,p
            # * We wish to group by ll,mm,l,m,n and thus leave p to vary "naturally" over the domain
            tmp_qnmc = this.data['per_qnm'].keys()
            # Strip away the p coordinate. Get unique coordinates, and sort for aesthetics
            #                          ( (ll,mm), ( l    ,  m    , n, ...  )
            qnmc = sorted( list( set([ (  c[0]  , tuple([d for k,d in enumerate(c[1]) if not (k in [3,7]) ]) )  for c in tmp_qnmc ]) ) )

            # #
            # select_qnmc = [ ((2,2),(2,2,0)),
            #                 ((2,2),(2,2,1)),
            #                 ((2,1),(2,1,0)),
            #                 ((3,3),(3,3,0)),
            #                 ((3,3),(3,3,1)),
            #                 ((3,2),(3,2,1)),
            #                 ((4,4),(4,4,0)),
            #                 ((4,3),(4,3,0)),
            #                 ((4,4),(2,2,0,2,2,0)),
            #                 ((4,3),(2,2,0,2,1,0)) ]

            #
            select_qnmc = [ ((2,2),(2,2,0)),
                            ((2,2),(2,2,1)),
                            ((2,1),(2,1,0)),
                            ((3,3),(3,3,0)),
                            ((3,3),(3,3,1)),
                            ((3,2),(3,2,0)),
                            ((3,2),(2,2,0)),
                            ((4,4),(4,4,0)),
                            ((4,3),(4,3,0)),
                            ((4,3),(3,3,0)) ]

            # NOTE that we will only work with the desired QNM fits
            qnmc = [ c for c in qnmc if c in select_qnmc  ]

            # ---------------------------------------- #
            # For all QNM coordinates, model and store to current object
            # ---------------------------------------- #
            for jk_ in qnmc:

                # Let the people know
                pad = '**'*25
                print '\n%s\n*\tNow Modeling: %s\n%s' % (pad,green(str(list(jk_))),pad)
                sys.stdout.flush()

                # Collect domain values
                eta = this.list_qnm_data( jk_,'eta' )
                chi1 = this.list_qnm_data( jk_,'chi1' )
                chi2 = this.list_qnm_data( jk_,'chi2' )
                m1 = this.list_qnm_data( jk_,'m1' )
                m2 = this.list_qnm_data( jk_,'m2' )

                # Calculate transformed domain coordinates
                chi_s = ( m1*chi1 + m2*chi2 )/( m1+m2 )
                chi_a = ( m1*chi1 - m2*chi2 )/( m1+m2 )
                dM_ = ( m1-m2 ) / ( m1+m2 )
                chi = chi_s + dM_*chi_a - eta*chi_s*76.0/113.0

                # Define a holder for general domain data
                bulk_domain = { 'eta':eta,
                                'dM':dM_,
                                'chi_s':chi_s,
                                'chi_a':chi_a,
                                'chi':chi
                              }

                # # Define a bulk dictionary space to use for the positive greedy process
                # bulk = bulk_domain.keys()
                #
                # # Define an action for the greedy process
                # def action( trial_boundary ):
                #     domain = make_domain( trial_boundary )
                #     print '>> Domain shape is %s'%list(domain.shape)
                #     scalar_range = ak
                #     foo = gmvpfit( domain, scalar_range, fitatol=this.fitatol )
                #     estimator = foo.frmse
                #     return estimator,foo
                #
                # # Given the action and the bulk, apply a positive greedy algorithm
                # A = pgreedy( bulk, action, fitatol=fitatol, verbose=verbose  )

                # ---------------------------------------- #
                # Construct labels for model entities
                # ---------------------------------------- #
                m = jk_[1][1]
                python_prefix = 'eta*sqrt(1-4.0*eta)' if mod(m,2) else 'eta'
                latex_prefix = r'\eta \, \sqrt{1-4\eta}' if mod(m,2) else r'\eta'
                if 3 == len(jk_[1]):
                    # Handle first order modes
                    (ll_,mm_),(l_,m_,n_) = jk_[0],jk_[1]
                    range_label = 'A%i%i%i%i%i'%(ll_,mm_,l_,m_,n_)
                    domain_label = tuple( this.model_domain_variables )
                    latex_range_label = r'A_{%i%i%i%i%i}'%(ll_,mm_,l_,m_,n_)
                    latex_domain_label = (r'\%s'%this.model_domain_variables[0],r'\%s'%this.model_domain_variables[1])
                    labels = { 'python':[range_label,domain_label,python_prefix], 'latex':[latex_range_label,latex_domain_label,latex_prefix] }
                    # Store azimulathal eigenvalues for future reference
                    mm,meff = mm_,m_
                else:
                    # Handle 2nd order modes
                    (ll_,mm_),(l1_,m1_,n1_,l2_,m2_,n2_) = jk_[0],jk_[1]
                    range_label = 'A%i%i%i%i%i%i%i%i'%(ll_,mm_,l1_,m1_,n1_,l2_,m2_,n2_)
                    domain_label = tuple( this.model_domain_variables )
                    latex_range_label = r'A_{%i%i%i%i%i%i%i%i}'%(ll_,mm_,l1_,m1_,n1_,l2_,m2_,n2_)
                    latex_domain_label = (r'\%s'%this.model_domain_variables[0],r'\%s'%this.model_domain_variables[1])
                    labels = { 'python':[range_label,domain_label,python_prefix], 'latex':[latex_range_label,latex_domain_label,latex_prefix] }
                    # Store azimulathal eigenvalues for future reference
                    mm,meff = mm_,m1_+m2_

                # ---------------------------------------- #
                # Collect range values: NOTE that Ak below are phase aligned strain values
                # ---------------------------------------- #
                Ak = this.get_phase_aligned_strain_amplitudes(jk_)

                # Define a function to make domains given a list of desired domain names
                make_domain = lambda domain_name_list: vstack( [ bulk_domain[name] for name in domain_name_list ] ).T

                # Define domain and range to model
                domain = make_domain( this.model_domain_variables )
                scalar_range = Ak

                # ---------------------------------------- #
                # Model select modes
                # ---------------------------------------- #
                # Retrieve range map for this mode
                range_map = this.get_range_map(jk_)
                # Define selection criteria
                the_space_is_well_posed = domain[isfinite(scalar_range),:].shape[-1] < domain[isfinite(scalar_range),:].shape[0]
                the_qnm_is_not_drift = mm == meff
                # Impose selection criteria
                if the_space_is_well_posed and the_qnm_is_not_drift:
                    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
                    foo = gmvpfit( domain, scalar_range, fitatol=this.fitatol,
                                   labels=labels, verbose=this.verbose, maxdeg=6,
                                   mindeg=sign(mm)*max(abs(mm)-1,2), range_map=range_map )
                    # IMPORTANT NOTE(s)
                    #   * Do not set initial_boundary=['K'] here. This will
                    #     generally muck things up. *_*
                    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
                    sys.stdout.flush()
                else:
                    foo = None
                    if the_qnm_is_not_drift:
                        alert('This mode will not be modeled becuase it does not pass the current selection criteria: m_eff = m1+m2 (azimuthal m, not mass)')

                # Store the model to the current object
                this.models[ jk_ ] = foo

            # Write the models to disk
            if this.verbose: print '%s\n# Saving model data to file: "%s"\n%s'%('####'*12,cyan(models_data_path),'####'*12)
            sys.stdout.flush()
            with open(models_data_path, 'wb') as models_file:
                pickle.dump( this.models , models_file, pickle.HIGHEST_PROTOCOL )
            if this.verbose: alert(' We may now Proceed to documenting.')

    # Filter scentry list by resolution
    def filter_scentry_resolution( this, scentry_iterable, res_min=160 ):
        '''Filter scentry list by resolution'''
        # Let the people know
        if this.verbose: alert('Now filtering scentry iterable by resolution (GT only) ...')
        # Create a boolean mask to apply to the scentry iterable
        mask = []
        # For all simulations
        reject_list = [ 'um4_88','q10c25e1c_T_80_320', 'q18a0a03c05v4_64_128','q18a0aM03c05v4_64_128', 'q18a0aM04c05v4_et05_2_T_80', 'q4a025a05_T_80_320_ERit2', 'q4a075a0375_T_80_320_ERit2_AH_opt_C0.4', 'q4aM025a0625_T_80_320_ERit2', 'q8_T_64_336', 'D10_q3.00_a0.8_0.0_m240', 'D12_q2.00_a0.15_-0.60_m160' ]
        for e in scentry_iterable:

            # If the simulation is of the gt institute
            if e.config.institute.lower() == 'gt':
                # Extract resolution parameter string
                res_string =  [ k[1:] for k in e.simname.lower().split('_') if k[0]=='m' ][0] if '_m' in e.simname.lower() else 'None'
                # If its numeric, compare it to the threshold
                res = int(res_string) if res_string.isdigit() else res_min-1
                # Determine whether to keep the simulation
                keep = (res >= res_min)
            else:
                # We will only work with GT waveforms at the moment, all others will be kept without inspection
                keep = True

            #
            keep = keep and not ( e.simname in reject_list )
            # Let the people know
            if keep:
                if this.verbose: alert('Keeping "%s"'%green(e.simname))
            else:
                if this.verbose: alert('Removing "%s"'%magenta(e.simname))

            # Append the mask
            mask.append(keep)
        # Apply the mask
        filtered_scentry_iterable = [ e for k,e in enumerate(scentry_iterable) if mask[k] ]
        # Return the answer
        ans = filtered_scentry_iterable
        return ans

    # Document modeling results, low-level: python formulas and latex
    def document(this,plot=True,latex=True,python=True):

        # Import useful thing(s)
        from os import system as bash

        # Let the people know
        pad = '####'*12+'#'
        if this.verbose: print '%s\n# Documenting the model results: LaTeX formula, (todo: plots)\t\t#\n%s'%(pad,pad)

        # Define output dirs
        docdir = this.workdir+'docs/'

        # Dir to hold python modules
        texdir = docdir + 'latex/' + this.keyword + '/'

        # File to hold python rep of models
        pyfile = docdir + 'mmrdnp' + ('' if this.keyword is None else '_') + this.keyword + '.py'

        # Create output directories
        # NOTE that all parents will be created
        mkdir(texdir,verbose=True)

        # Store T0 value to file forr external referencing
        texfile = texdir + 'T0.tex'
        fid = open( texfile, 'w+' )
        fid.write( r'\def\T0{\check{%i}}%s'%(this.T0,'\n') )
        fid.close()

        # Organize fits by ll,mm
        fit = {}
        for jk_ in this.models:
            # Extract spherical indeces
            (ll,mm) = jk_[0]
            # Initiate key
            if not ( (ll,mm) in fit ):
                fit[ll,mm] = []
            # Append to list for the ll,mm
            if this.models[jk_] is not None:
                fit[ll,mm].append( this.models[jk_] )

        # For each ll,mm open a file in the latex directory for writing
        if latex:
            for ll,mm in fit:

                # Define the file string and open the file
                texfile = texdir + 'l%im%i.tex' % (ll,mm)
                fid = open( texfile, 'w+' )

                #
                print '>> Outputting LaTeX for (ll,mm) = (%i,%i)'%(ll,mm)

                # For each fit of ll,mm write the latex string as well as a comment with useful information such as frmse, and domain size
                for k,f in enumerate(fit[ll,mm]):
                    #
                    if (f.frmse < 1) and (f.frmse > 1e-8) :

                        # Latex comment
                        comment = '  %% QNM Strain Amplitude. FRMSE = %f. Domain shape: %s'%(f.frmse,list(f.domain.shape)) + '\n'
                        # Equation label
                        eqlabel = '  \\label{eq:A%i%i_fit_%i}\n'%(ll,mm,k+1)
                        # Equation
                        equation = '  '+f.__str_latex__(precision=4) + ( r'  \\' + '\n' if k+1 < len(fit[ll,mm]) else ' ' )

                        #
                        fid.write(comment)
                        fid.write(eqlabel)
                        fid.write(equation)

                #
                fid.close()



        # For each ll,mm open a file in the latex directory for writing
        if python:

            #
            fid = open( pyfile, 'w+' )

            #
            print '>> Outputting Python for (ll,mm) = (%i,%i)'%(ll,mm)

            #
            pad = '\t\t\t  '
            fid.write('# Import useful things\n')
            fid.write('from numpy import exp,sqrt\n')
            fid.write('\n')
            fid.write('# Define a dictionary to hold the lambda function for each QNM\n')
            fid.write('A_strain_np = {\n%s'%pad)

            counter = 0
            jk_list = sorted( list( this.models.keys() ), key=lambda x: x[0][0] )
            for jk_ in jk_list:

                # For each fit of ll,mm write the latex string as well as a comment with useful information such as frmse, and domain size
                f = this.models[jk_]

                # Equation label
                eqkey = '%s'%str(jk_)
                # Equation
                equation = f.__str_python__(precision=12).split('=')[-1]

                #
                counter += 1
                if counter < len(this.models):
                    fid.write(eqkey+':'+equation+',\n%s'%pad)
                else:
                    fid.write(eqkey+':'+equation+'\n%s'%pad)

            fid.write('}\n')

            #
            fid.close()

            #
            formula_dir = '/Users/book/KOALA/kerr_dev/kerr/formula/'
            bash('cp %s %s'%(pyfile,formula_dir))

        #
        # modes_to_plot = this.models.keys()
        modes_to_plot = [ ( (2,2),(2,2,0) ),
                          ( (2,1),(2,1,0) ),
                          ( (3,3),(3,3,0) ),
                          ( (3,3),(3,3,1) ),
                          ( (3,2),(3,2,0) ),
                          ( (3,2),(2,2,0) ),
                          ( (4,4),(4,4,0) ),
                          ( (4,3),(4,3,0) ),
                          ( (4,3),(3,3,0) )]
        # modes_to_plot = [ ( (2,2),(2,2,0) ) ]

        #
        if plot:
            for jk_ in modes_to_plot:
                #
                print '>> Plotting Ak 2D Countour for %s'%list(jk_)
                this.plot2DSurf(jk_)
                #
            for lm in ( (2,2), (3,3) ):
                #
                print '>> Plotting ysprod 2D Countour for %s'%list(lm)
                this.plotYSProdRatio(lm)

    # Plot 2D contours (or graddients) of model fits
    def plot2DSurf(this,jk_):
        '''Plot 2D contours (or graddients) of model fits '''

        # Import usefult things
        from matplotlib.pyplot import figure,plot,scatter,xlabel,ylabel,savefig,imshow,colorbar,gca
        from numpy import linspace,meshgrid,array,angle,unwrap
        from matplotlib import cm

        # Extract the model for the desired ( (ll,mm), (l,mn) )
        f = this.models[jk_]

        # Convert the QNM coordinate to a string
        def jk2str(jk):
            rep = [' ','[',']','(',')',',']; mid = str(list(jk))
            for r in rep: mid = mid.replace(r,'')
            return mid
        #
        mid = jk2str(jk_)

        # Define number of points to use
        N = 200
        # Define coordinate ranges
        eta_min,eta_max,chi_min,chi_max = 0,0.25,-1,1
        eta_range = linspace(0,0.25,N)
        chi_range = linspace(-1,1,N)

        # Create a grid of the coordinate ranges
        X,Y = meshgrid( eta_range,chi_range )
        domain,_ = ndflatten( [X,Y], Y )
        scalar_range = f.eval(domain)
        SR = scalar_range.reshape( X.shape )

        #
        kind_list = ['amp','phase']

        # Define output directory for plots
        outdir = '/Users/book/KOALA/kerr_dev/tex/np/fig/%s/'%this.keyword
        mkdir(outdir,verbose=True)

        # Create plot for Amplitude and Phase separately
        for kind in kind_list:

            #
            fig = figure( figsize=this.figsize )

            #
            ax = splot( domain, SR, f.domain, f.range, kind=kind )

            #
            ax = gca()
            ax.grid(True)
            gridlines = ax.get_xgridlines() + ax.get_ygridlines()
            for line in gridlines:
                line.set_linestyle(':')
                line.set_color('k')
                line.set_alpha(0.25)
            ax.set_aspect(0.12)
            ax.set_xlabel('$\%s$'%this.model_domain_variables[0])
            ax.set_ylabel('$\%s$'%this.model_domain_variables[1])
            xticks = [ round(k,2) for k in linspace( 0,0.25,6 )[1:] ]
            ax.set_xticks( xticks )

            #
            ttl = '$|A_{%s}|$'%mid  if kind=='amp' else '$\mathrm{arg}(A_{%s})$'%mid
            ax.set_title(ttl)

            #
            name = 'A'+mid+'_amp' if kind=='amp' else 'A'+mid+'_phase'
            savefig( outdir+name+'.pdf', bbox_inches='tight', pad_inches=0 )

    # Plot ysprod ratio Comparison
    def plotYSProdRatio(this,lm):
        '''Given l and m, calculate inner product ration:
            rho = <y,s>_(l+1,m) / <y,s>_(l,m)
           Where y is a sphericcal harmonic and s is a
           spheroidal harmonic.
        '''

        # Import useful things
        from nrutils import FinalSpin0815,EradRational0815
        from numpy import array,linspace,meshgrid,vstack,ones,sqrt
        from kerr import ysprod
        from matplotlib.pyplot import figure,plot,scatter,xlabel,ylabel,savefig,imshow,colorbar,gca
        # Setup plotting backend
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20

        #
        l,m = lm

        # Name full QNM indeces of interest using inputs
        jk_1 = ( (l,m),(l,m,0) )
        jk_2 = ( (l+1,m),(l,m,0) )

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # Calculate the inner product ratios from the NR data
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        # Name variables of intertest
        var_name = [ 'Ak', 'eta', 'chi1', 'chi2', 'jf', 'Mf', 'sim_id', 'm1', 'm2' ]

        # Extract useful data from each mode coordinate
        data1 = { var:this.list_qnm_data( jk_1,var ) for var in var_name }
        data2 = { var:this.list_qnm_data( jk_2,var ) for var in var_name }

        # Find data from like simulations
        def dintersect( A1,A2,sid1,sid2 ):
            a = sid2 if len(sid2)>len(sid1) else sid1
            b = sid1 if a is sid2 else sid2
            mask = array( [ (k in b) for k in a ] )
            A2 = A2[mask] if a is sid2 else A2
            A1 = A1[mask] if a is sid1 else A1
            return A1,A2

        # Find the common dataset in simulation ID
        for var in [ v for v in var_name if v!='sim_id' ]:
            data1[var],data2[var] = dintersect( data1[var],data2[var], data1['sim_id'],data2['sim_id'] )

        #
        chi_s = ( data1['chi1']*data1['m1'] + data1['chi2']*data1['m2'] ) / ( data1['m1']+data1['m2'] )
        nr = { 'domain':vstack( [ data1['eta'], chi_s ] ).T, 'range':data2['Ak']/data1['Ak'] }

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # Calculate the inner product ratios for ideal data (i.e. from perturbation theory)
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        # Define number of points to use
        N = round( sqrt(400) )
        # Define coordinate ranges
        eta_min,eta_max,chi_min,chi_max = 0,0.25,-1,1
        eta_range = linspace(0,0.25,N)
        chi_range = 0.9*linspace(-1,1,N)
        ETA,CHI = meshgrid( eta_range,chi_range )
        domain = ndflatten( [ETA,CHI] )
        #
        scalar_range = array( [ FinalSpin0815(domain[k,0],domain[k,1],0) for k in range(len(domain[:,0])) ] )
        #
        print '>> Calculating analytic ysprods for %s '%magenta('(l,m)=(%i,%i)'%(l,m)),
        ptysp1 = ones( scalar_range.shape, dtype=complex )
        ptysp2 = ones( scalar_range.shape, dtype=complex )
        for k,jf in enumerate( scalar_range ):
            print '.',
            #
            ll,mm = jk_1[0]; l,m,n = jk_1[1];
            ptysp1[k] = ysprod( jf, ll, mm, (l,m,n) )
            #
            ll,mm = jk_2[0]; l,m,n = jk_2[1];
            ptysp2[k] = ysprod( jf, ll, mm, (l,m,n) )
        print cyan('=\(^_^)/=')

        #
        pt = { 'domain':domain, 'range': (ptysp2/ptysp1).reshape(ETA.shape) }

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # Plot the Comparison
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        #
        kind_list = [ 'amp','phase' ]

        # Define output directory for plots
        outdir = '/Users/book/KOALA/kerr_dev/tex/np/fig/%s/'%this.keyword
        mkdir(outdir,verbose=True)

        #
        for kind in kind_list:

            #
            fig = figure( figsize= 0.4*this.figsize*array([1,0.95]) )

            #
            ax = splot(pt['domain'],pt['range'],nr['domain'],nr['range'],kind=kind,ms=80,cbfs=17)
            #
            ax.grid(True)
            gridlines = ax.get_xgridlines() + ax.get_ygridlines()
            for line in gridlines:
                line.set_linestyle(':')
                line.set_color('k')
                line.set_alpha(0.25)
            ax.set_aspect(0.12)
            ax.set_xlabel('$\%s$'%this.model_domain_variables[0])
            ax.set_ylabel('$\%s$'%this.model_domain_variables[1])
            xticks = [ round(k,2) for k in linspace( 0,0.25,6 )[1:] ]
            ax.set_xticks( xticks )
            prefix = r'|\rho|' if kind is 'amp' else r'\mathrm{arg}(\rho)'
            rho = r'\rho_{%i%i%i%i}^{%i%i%i}'%(tuple( list(jk_1[0])+list(jk_2[0])+list(jk_1[1])) )
            ttl = r'$|%s|$'%rho if kind=='amp' else r'$\mathrm{arg}(%s)$'%rho
            ax.set_title(ttl,y=1.01)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(19)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(24)

            #
            prefix = 'rho_%i%i%i%i_%i%i%i'%(tuple( list(jk_1[0])+list(jk_2[0])+list(jk_1[1])) )
            name = prefix+'_amp' if kind=='amp' else prefix+'_phase'
            savefig( outdir+name+'.pdf', bbox_inches='tight', pad_inches=0 )


# Least squares exponential
def lsexp( y, t, cwlist ):

    # Import useful things
    from numpy import exp,vstack,dot,array
    from numpy.linalg import pinv,lstsq

    #
    Z = vstack( [ exp(1j*cw.conj()*t) for cw in cwlist ] ).T

    #
    V = pinv( Z )

    #
    a = dot( V, y )

    #
    return a,array(cwlist)



# ############################################################ %
''' Greedy Algorithm with allowed non-optimal steps '''
# ############################################################ %
def gnx(y,qnminfo,verbose=False):

    error('this function isnt ready to be used yet -- still translating from matlab version')

    #
    from numpy import inf,array,amin,amax,exp

    #
    y_input = y
    y = y_norm(y_input)
    phi0 = y.phi_raw(1)
    y = y_rotate(y,phi0)
    y.t_raw = y.t_raw - y.t_raw(1)
    y.w = 2*pi*y.f

    # ------------------------------------------------- %
    # Extract perturbation frequencies from qnm_info
    # ------------------------------------------------- %
    all_w = qnminfo.cw

    # Always include l=ll m=mm QNM
    # ------------------------------------------------- %
    wllmm = []
    if not NOHOME:
        map_ = qnm_info.search( 'id', qnm_info.child(1).calc_id([ y.l, y.m, 0, 1 ]) )
        wllmm = qnm_info.child(map_).complex_w

    # Remove these modes from consideration
    for i in range(len(wllmm)):
        all_w = all_w( all_w != wllmm(i) )
    # Make sure that the list only contains unique frequencies
    all_w = list(set(all_w))

    # ------------------------------------------------- %
    # Optimize over presumed QNM content!
    # ------------------------------------------------- %
    if verbose:
        if not STATIC:
            print('[y_depict.m::gnx]>> Using greedy algorithm to optimize over presumed QNM content...\n')

    # GREEDY NODD
    done = False; rmse_best = inf;
    i = 0; w_space = all_w; kept_w = wllmm;
    too_full_counter = 0
    stat_t_max = 15     # Larget time to the right of y.t_raw(1) to use for statistics
    too_full_max = 2    # Allowing this value to be too large causes the addition of
                        # incidental modes to be added - e.g. ones that aren't physical,
                        # but make the fit better.
    while not done:

        i = i+1;

        too_full_max = min(too_full_max,length(w_space))
        rmse = ones(size(w_space))
        trial = cell(size(w_space))
        for j in range(len(w_space)):

            # create list of frequencies to use
            try:
                wvals = list( kept_w )
                wvals.append( w_space[j] )
            except:
                print '??'
                raise

            amplitude,complex_w = nodd_helper(y,wvals)
            trial[j] = package_output(complex_w,amplitude)

            trial[j] = nodd_statistics( trial[j], y, stat_t_max )
            rmse[j] = trial[j].stats.mean_rmse


        # find the best one, add to greedy list
        rmse[rmse == 1] = 100*rmse[rmse==1] # make sure that invalid cases are ignored
        [min_rmse,min_map] = amin(rmse)

        if isnan(min_rmse):
            raise

        # if rmse is less than min_rmse, keep this mode information for
        # output
        if min_rmse < rmse_best: # (1-min_rmse/rmse_best) > 1e-2 # min_rmse < rmse_best
            kept_w.append( w_space[min_map] )
            w_space[min_map] = []
            x = trial[min_map] # store this step for output
            rmse_best = min_rmse
            too_full_counter = 0 # reset "too full counter"

            if verbose: print '[%i%i] *min_rmse = %1.5f, rmse_best = %1.5f\n'%(too_full_counter,too_full_max,min_rmse,rmse_best)
        else: # keep this mode information, but continue processing greedily
            too_full_counter = 1 + too_full_counter
            too_full = (too_full_counter >= too_full_max) or (len(w_space)==1);
            if too_full: # if too many greedy steps have been taken with no pay-off, stop the algorith
                done = true
            else: # else, keep being greedy
                kept_w.append( w_space[min_map] )
                w_space[min_map] = []
            if verbose: print '[%i,%i] min_rmse = %1.5f, rmse_best = %1.5f\n'%(too_full_counter,too_full_max,min_rmse,rmse_best)


    # Rescale amplitudes and reshift phases
    if rmse_best>1-1e-2:
        K = 1e-10
    else:
        K = 1
    x.amplitude = x.amplitude * y.norm_const
    x.amplitude = K * x.amplitude * exp(-1j*phi0)
    x.rmse = rmse_best

    # # Given the solution above, estimate the mean and std of the ampltides
    # # by condiering 10 points within 10*(time units) to the right, assuming
    # # that rescaling by the first solution holds.
    # x = nodd_statistics(x,y,stat_t_max)


    if verbose: print('\n')
