
# Lentz's continued fration solver
def lentz( aa, bb, tol=1e-10, tiny=1e-30, mpm=False ):
    '''
    Lentz's method for accurate continued fraction calculation of a function
    f:

    f = b(0) + a(1)/( b(1) + a(2)/( b(2) + a(3)/( b(3) + ... ) ) )

    (Equivalent notation)

    f = b(0) + [a(1)/b(1)+][a(2)/b(2)+][a(3)/b(3)+]...[a(n)/b(n)+]...

    References:

    http://www.mpi-hd.mpg.de/astrophysik/HEA/internal/Numerical_Recipes/f5-2.pdf
    http://epubs.siam.org/doi/pdf/10.1137/1.9780898717822.ch6

    ~ llondon6'12
    [CONVERTED TO PYTHON FROM MATLAB by llondon2'14]
    '''

    #
    f = bb(0)
    if 0==f: f = tiny

    if mpm:
        from mpmath import mpc
        C,D = mpc(f),mpc(0)
    else:
        from numpy import complex256
        C,D = complex256(f),complex256(0)

    done,state = False,False
    j,jmax = 0,2e3
    while not done:

        #
        j = 1+j

        #
        D = bb(j) + aa(j)*D
        if 0==D: D = tiny
        #
        C = bb(j) + aa(j)/C
        if 0==C: C = tiny
        #
        D = 1.0/D
        DELTA = C*D
        #
        f = f*DELTA
        #
        done = abs( DELTA - 1.0 )<tol
        if j>=jmax:
            # print('>>! Maximum number of iterations reached before error criteria passed.\n')
            state = True
            done = state

    return (f,state)


# Equation 27 of Leaver '86
def leaver27( a, l, m, w, A, s=-2.0, vec=False, mpm=False, **kwargs ):

    #
    pmax = 5e2

    #
    if mpm:
        from mpmath import sqrt
    else:
        from numpy import sqrt

    global c0, c1, c2, c3, c4, Alpha, Beta, Gamma, l_min

    #
    l_min = l-max(abs(m),abs(s)) # VERY IMPORTANT


    # ------------------------------------------------ #
    # Radial parameter defs
    # ------------------------------------------------ #
    #b  = complex(numpy.sqrt(1.0-4.0*a*a))
    b  = sqrt(1.0-4.0*a*a)
    c_param = 0.5*w - a*m

    c0    =         1.0 - s - 1.0j*w - (2.0j/b) * c_param
    c1    =         -4.0 + 2.0j*w*(2.0+b) + (4.0j/b) * c_param
    c2    =         s + 3.0 - 3.0j*w - (2.0j/b) * c_param
    c3    =         w*w*(4.0+2.0*b-a*a) - 2.0*a*m*w - s - 1.0 \
                    + (2.0+b)*1j*w - A + ((4.0*w+2.0j)/b) * c_param
    c4    =         s + 1.0 - 2.0*w*w - (2.0*s+3.0)*1j*w - ((4.0*w+2.0*1j)/b)*c_param

    Alpha = lambda k:	k*k + (c0+1)*k + c0
    Beta  = lambda k:   -2.0*k*k + (c1+2.0)*k + c3
    Gamma = lambda k:	k*k + (c2-3.0)*k + c4 - c2 + 2.0

    #
    v = 1.0
    for p in range(l_min+1):
        v = Beta(p) - ( Alpha(p-1.0)*Gamma(p) / v )

    #
    aa = lambda p:   -Alpha(p-1.0+l_min)*Gamma(p+l_min)
    bb = lambda p:   Beta(p+l_min)
    u,state = lentz(aa,bb)
    u = Beta(l_min) - u

    #
    x = v-u
    if vec:
        x = [x.real,x.imag]

    #
    return x


# Equation 21 of Leaver '86
def leaver21( a, l, m, w, A, s=-2.0, vec=False, **kwargs ):

    #
    pmax = 5e2

    global k1, k2, alpha, beta, gamma, l_min

    #
    l_min = l-max(abs(m),abs(s)) # VERY IMPORTANT

    # ------------------------------------------------ #
    # Angular parameter functions
    # ------------------------------------------------ #
    k1 = 0.5*abs(m-s)
    k2 = 0.5*abs(m+s)

    a=1.0*a
    alpha = lambda k:	-2.0 * (k+1.0) * (k+2.0*k1+1.0)
    beta  = lambda k:	k*(k-1.0) \
                        + 2.0*k*( k1+k2+1.0-2.0*a*w ) \
                        - ( 2.0*a*w*(2.0*k1+s+1.0)-(k1+k2)*(k1+k2+1) ) \
                        - ( a*w*a*w + s*(s+1.0) + A )
    gamma = lambda k:   2.0*a*w*( k + k1+k2 + s )

    #
    v = 1.0
    for p in range(l_min+1):
        v = beta(p) - (alpha(p-1.0)*gamma(p) / v)

    #
    aa = lambda p: -alpha(p-1.0+l_min)*gamma(p+l_min)
    bb = lambda p: beta(p+l_min)
    u,state = lentz(aa,bb)
    u = beta(l_min) - u

    #
    x = v-u
    if vec:
        x = [x.real,x.imag]

    #
    return x


# Work function for QNM solver
def leaver_workfunction( j, l, m, state, s=-2, mpm=False ):
    '''
    work_function_to_zero = leaver( state )

    state = [ complex_w complex_eigenval ]
    '''

    #
    from numpy import complex128
    if mpm:
        import mpmath
        mpmath.mp.dps = 8
        dtyp = mpmath.mpc
    else:
        from numpy import complex256 as dtyp

    # Unpack inputs
    a = dtyp(j)/2.0                 # Change from M=1 to M=1/2 mass convention

    #
    complex_w = 2.0*dtyp(state[0])  # Change from M=1 to M=1/2 mass convention
    ceigenval = dtyp(state[1])

    #
    if len(state) == 4:
        complex_w = 2 * (dtyp(state[0])+1.0j*dtyp(state[1]))
        ceigenval = dtyp(state[2]) + 1.0j*dtyp(state[3])

    # concat list outputs
    x = leaver21(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm) +  leaver27(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm)

    #
    x = [ complex128(e) for e in x ]

    #
    return x

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Implement Berti's approximation for the separation constants '''
# NOTE that Beuer et all 1977 did it first!?
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def scberti(c,          # dimensionless cw*jf
            l,m,s=-2):

    #
    from numpy import zeros,array

    # NOTE that the input here is c = jf*complex_w
    f = zeros((6,),dtype='complex128')

    #
    l,m,s = float(l),float(m),float(s)

    f[0] = l*(l+1) - s*(s+1)
    f[1] = - 2.0 * m * s*s / ( l*(l+1) )

    hapb = max( abs(m), abs(s) )
    hamb = m*s/hapb
    h = lambda ll: (ll*ll - hapb*hapb) * (ll*ll-hamb*hamb) * (ll*ll-s*s) / ( 2*(l-0.5)*ll*ll*ll*(ll-0.5) )

    f[2] = h(l+1) - h(l) - 1
    f[3] = 2*h(l)*m*s*s/((l-1)*l*l*(l+1)) - 2*h(l+1)*m*s*s/(l*(l+1)*(l+1)*(l+2))
    f[4] = m*m*s*s*s*s*( 4*h(l+1)/(l*l*(l+1)*(l+1)*(l+1)*(l+1)*(l+2)*(l+2)) \
                        - 4*h(l)/((l-1)*(l-1)*l*l*l*l*(l+1)*(l+1)) ) \
                        - (l+2)*h(l+1)*h(l+2)/(2*(l+1)*(2*l+3)) \
                        + h(l+1)*h(l+1)/(2*l+2) + h(l)*h(l+1)/(2*l*l+2*l) - h(l)*h(l)/(2*l) \
                        + (l-1)*h(l-1)*h(l)/(4*l*l-2*l)

    '''
    # NOTE that this term diverges for l=2
    f[5] = m*m*m*s*s*s*s*s*s*( 8.0*h(l)/(l*l*l*l*l*l*(l+1)*(l+1)*(l+1)*(l-1)*(l-1)*(l-1)) \
                             - 8.0*h(l+1)/(l*l*l*(l+1)*(l+1)*(l+1)*(l+1)*(l+1)*(l+1)*(l+2)*(l+2)*(l+2)) ) \
              + m*s*s*h(l) * (-h(l+1)*(7.0*l*l+7*l+4)/(l*l*l*(l+2)*(l+1)*(l+1)*(l+1)*(l-1)) \
                              -h(l-1)*(3.0*l-4)/(l*l*l*(l+1)*(2*l-1)*(l-2)) ) \
              + m*s*s*( (3.0*l+7)*h(l+1)*h(l+2)/(l*(l+1)*(l+1)*(l+1)*(l+3)*(2*l+3)) \
                                 -(3.0*h(l+1)*h(l+1)/(l*(l+1)*(l+1)*(l+1)*(l+2)) + 3.0*h(l)*h(l)/(l*l*l*(l-1)*(l+1)) ) )
    '''

    # Calcualate the series sum, and return output
    return sum( f * array([ c**k for k in range(len(f)) ]) )


# ------------------------------------------------------------------ #
# Calculate the inner-product between a spherical and spheroidal harmonic
# ------------------------------------------------------------------ #
def ysprod( jf,
            ll,
            mm,
            lmn,
            N=2**9,         # Number of points in theta to use for trapezoidal integration
            theta = None,   # Pre computed theta domain
            verbose=False):

    #
    from kerr.pttools import slm
    from kerr import sYlm as ylm
    from kerr import warning
    from numpy import pi,linspace,trapz,sin,sqrt

    #
    th = theta if not (theta is None) else linspace(0,pi,N)
    ph = 0

    #
    prod = lambda A,B: 2*pi * trapz( A.conj()*B*sin(th), x=th )

    # Validate the lmn input
    if len(lmn) not in (3,6):
        error('the lmn input must contain only l m and n; note that the p label is handeled here by the sign of jf')

    # Unpack 1st order mode labels
    if len(lmn)==3:
        l,m,n = lmn
        so=False
        m_eff = m
    elif len(lmn)==6:
        l,m,n,_,l2,m2,n2,_ = lmn
        so=True
        m_eff = m+m2
        if verbose: warning('Second Order mode given. An ansatz will be used for what this harmonic looks like: products of the related 1st order spheroidal functions. This ansatz could be WRONG.','ysprod')


    #
    if m_eff==mm:
        #
        y = ylm(-2,ll,mm,th,ph)
        _s = slm(jf,l,m,n,th,ph,norm=False,__rescale__=False) if not so else slm(jf,l,m,n,th,ph,norm=False,__rescale__=False)*slm(jf,l2,m2,n2,th,ph,norm=False,__rescale__=False)
        s = _s / sqrt(prod(_s,_s))
        ans = prod( y,s ) # note that this is consistent with the matlab implementation modulo the 2*pi convention
    else:
        # print m,m_eff,mm,list(lmnp)
        ans = 0
        # raise

    return ans

# ------------------------------------------------------------------ #
# Calculate inner product of two spheroidal harmonics at a a given spin
# NOTE that this inner product does not rescale the spheroidal functions so that the spherical normalization is recovered
# ------------------------------------------------------------------ #
def ssprod( jf, z1, z2, verbose=False, prod=None, N=2**9 ):

    #
    from numpy import linspace,trapz,array,pi,sin

    #
    l1,m1,n1 = z1
    l2,m2,n2 = z2

    #
    if m1 == m2 :
        #
        th, phi = pi*linspace(0,1,N), 0
        # Handle optional inner product definition
        if prod is None:
            prod = lambda A,B: 2*pi * trapz( A.conj()*B*sin(th), x=th )
        #
        s1 = slm( jf, l1, m1, n1, th, phi, norm=False, __rescale__=False )
        s2 = slm( jf, l2, m2, n2, th, phi, norm=False, __rescale__=False ) if (l2,m2,n2) != (l1,m1,n1) else s1
        #
        ans = prod(s1,s2)
    else:
        ans = 0

    #
    return ans

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Spheroidal Harmonic angular function via leaver's sum '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def slm(  jf,               # Dimentionless spin parameter
          l,
          m,
          n,
          theta,            # Polar spherical harmonic angle
          phi,              # Azimuthal spherical harmonic angle
          s       = -2,     # Spin weight
          plot    = False,  # Toggel for plotting
          __rescale__ = True, # Internal usage only: Recover scaling of spherical harmonics in zero spin limit
          norm = True,     # If true, normalize the waveform
          ax = None,        # axes handles for plotting to; must be length 1(single theta) or 2(many theta)
          verbose = False ):# Be verbose

    # Setup plotting backend
    if plot:
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.labelsize'] = 16

    #
    from kerr import leaver as lvr
    from kerr.formula.ksm2_slm_norm import CC as normcfit
    from kerr import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,ssprod
    from numpy import complex256,cos,ones,mean,isinf,pi,exp,array,ndarray,unwrap,angle,linalg,sqrt,linspace,sin,float128
    from matplotlib.pyplot import subplot,gca,xlabel,ylabel,xlim,ylim,title,figure,sca
    from matplotlib.pyplot import plot as plot_
    from scipy.misc import factorial as f
    from scipy.integrate import trapz

    #
    s = -2

    # Define an absolute error tolerance
    et = 1e-8

    # Use tabulated cw and sc values from the core package
    cw,sc = lvr( jf, l, m, n )

    # Ensure that thet is iterable
    if not isinstance(theta,ndarray):
        theta = array([theta])
    # Validate the QNM frequency and separation constant used
    lvrtol=1e-4
    lvrwrk = linalg.norm( leaver_workfunction(jf,l,m,[cw.real,cw.imag,sc.real,sc.imag]) )
    if lvrwrk>lvrtol:
        msg = 'There is a problem in '+cyan('kerr.core.leaver')+'. The values output are not consistent with leaver''s characteristic equations within %f.\n%s\n# The mode is (jf,l,m,n)=(%f,%i,%i,%i)\n# The leaver_workfunction value is %s\n%s\n'%(lvrtol,'#'*40,jf,l,m,n,red(str(lvrwrk)),'#'*40)
        error(msg,'slm')
    # If verbose, check the consisitency of the values used
    if verbose:
        msg = 'Checking consistency of QNM frequncy and separation constant used against Leaver''s constraint equations:\n\t*  '+cyan('leaver_workfunction(jf=%1.4f,l,m,[cw,sc]) = %s'%(jf,lvrwrk))+'\n\t*  cw = %s\n\t*  sc = %s'%(cw,sc)
        alert(msg,'slm')

    # Validate the spin input
    if isinstance(jf,int): jf = float(jf)

    # Define dimensionless deformation parameter
    aw = complex256( jf*cw )

    # ------------------------------------------------ #
    # Angular parameter functions
    # ------------------------------------------------ #
    k1 = 0.5*abs(m-s)
    k2 = 0.5*abs(m+s)
    alpha = lambda p:           -2.0*(p+1.0)*(p+2.0*k1+1.0)
    beta  = lambda p,A_lm,w_lm: p*(p-1.0)+2.0*p*(k1+k2+1.0-2.0*aw)\
                                -( 2.0*aw*(2.0*k1+s+1.0)-(k1+k2)*\
                                (k1+k2+1.0) ) - ( aw*aw + \
                                s*(s+1.0) + A_lm)
    gamma = lambda p,w_lm:      2.0*aw*(p+k1+k2+s)

    # ------------------------------------------------ #
    # Calculate the angular eighenfunction
    # ------------------------------------------------ #

    # Variable map for theta
    # u = float128( cos(theta) )
    u = float128( cos(theta) )

    # the non-sum part of eq 18
    X = ones(u.shape,dtype=complex256)
    X = X * exp(aw*u) * (1.0+u)**k1
    X = X * (1.0-u)**k2

    # initial series values
    a0 = 1.0 # a choice, setting the norm of Slm

    a1 = -beta(0,sc,cw)/alpha(0)

    C = 1.0
    C = C*((-1)**(max(-m,-s)))*((-1)**l)

    if True==norm:
        z = (l,m,n)
        C /= sqrt( ssprod(jf,z,z) )

    # Rescale to recover spherical harmonic limit at zero spin
    if __rescale__==True:
        thref = pi/2.5 # Some fiducial number where there are not likely to be roots
        phref = phi
        yref = sYlm(-2,l,m,thref,phref)
        sref = slm(0,l,m,n,thref,phref,__rescale__=False)
        # C = C*yref/sref
        if verbose:
            msg = 'The constant needed to recover the spherical harmonic scaling at zero spin is: %s'%(yref/sref)[0]
            alert(msg,'slm')

    # the sum part
    done = False
    Y = a0*ones(u.shape,dtype=complex256)
    Y = Y + a1*(1.0+u)
    k = 1
    kmax = 2e3
    err,yy = [],[]
    et2=1e-8
    max_a = max(abs(array([a0,a1])))
    while not done:
        k += 1
        j = k-1
        a2 = -1.0*( beta(j,sc,cw)*a1 + gamma(j,cw)*a0 ) / alpha(j)
        dY = a2*(1.0+u)**k
        Y += dY
        xx = max(abs( dY ))
        if plot:
            mask = abs(Y)!=0
            yy.append( C*array(Y)*X*exp(1j*m*phi) )
            err.append( xx )

        done = (k>=l) and ( (xx<et2 and k>30) or k>kmax )
        done = done or xx<et2
        a0 = a1
        a1 = a2

    # together now
    S = X*Y*exp(1j*m*phi)

    # Use same sign convention as spherical harmonics
    # e.g http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
    S = C * S

    #
    if plot:
        def ploterr():
            plot_( err, '-ob',mfc='w',mec='b' )
            # gca().set_xscale("log", nonposx='clip')
            gca().set_yscale("log", nonposx='clip')
            title( '$l = %i$, $m = %i$, $jf = %1.4f$'%(l,m,jf) )
            ylabel('Error Estimate')
            xlabel('Iteration #')
        if isinstance(theta,(float,int)):
            if ax is None:
                figure(figsize=3*array([3,3]))
            else:
                sca(ax[0])
            ploterr()
        elif isinstance(theta,ndarray):
            if ax is None:
                figure(figsize=3*array([6,2.6]))
                subplot(1,2,1)
            else:
                sca(ax[0])
            ploterr()
            if ax is not None:
                sca(ax[-1])
            else:
                subplot(1,2,2)
            clr = rgb( max(len(yy),1),reverse=True )
            for k,y in enumerate(yy):
                plot_(theta,abs(y),color=clr[k],alpha=float(k+1)/len(yy))
            plot_(theta,abs(S),'--k')
            pylim(theta,abs(S))
            fs=20
            xlabel(r'$\theta$',size=fs)
            ylabel(r'$|S_{%i%i%i}(\theta,\phi)|$'%(l,m,n),size=fs )
            title(r'$\phi=%1.4f$'%phi)

    #
    if len(S) is 1:
        return S[0]

    #
    return S
