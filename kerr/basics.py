
# -------------------------------------------------------- #
'''                 Define some useful BASICS            '''
# These are useful for terminal printing & system commanding
# -------------------------------------------------------- #

# Return name of calling function
def thisfun():
    import inspect
    return inspect.stack()[2][3]

# Make "mkdir" function for directories
def mkdir(dir_,rm=False,verbose=False):
    # Import useful things
    import os
    import shutil
    # Expand user if needed
    dir_ = os.path.expanduser(dir_)
    # Delete the directory if desired and if it already exists
    if os.path.exists(dir_) and (rm is True):
        if verbose:
            alert('Directory at "%s" already exists %s.'%(magenta(dir_),red('and will be removed')),'mkdir')
        shutil.rmtree(dir_,ignore_errors=True)
    # Check for directory existence; make if needed.
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        if verbose:
            alert('Directory at "%s" does not yet exist %s.'%(magenta(dir_),green('and will be created')),'mkdir')
    # Return status
    return os.path.exists(dir_)

# Alert wrapper
def alert(msg,fname=None,say=False,output_string=False):
    import os
    if fname is None:
        fname = thisfun()
    if say: os.system( 'say "%s"' % msg )
    _msg = '('+cyan(fname)+')>> '+msg
    if not output_string:
        print _msg
    else:
        return _msg

# Wrapper for OS say
def say(msg,fname=None):
    import os
    if fname is None:
        fname = thisfun()
    if msg:
        os.system( 'say "%s says: %s"' % (fname,msg) )

# Warning wrapper
def warning(msg,fname=None,output_string=False):
    if fname is None:
        fname = thisfun()
    _msg = '('+yellow(fname)+')>> '+msg
    if not output_string:
        print _msg
    else:
        return _msg

# Error wrapper
def error(msg,fname=None):
    if fname is None:
        fname = thisfun()
    raise ValueError( '('+red(fname)+')!! '+msg )

# Return the min and max limits of an 1D array
def lim(x):
    # Import useful bit
    from numpy import array,ndarray
    if not isinstance(x,ndarray):
        x = array(x)
    # Columate input.
    z = x.reshape((x.size,))
    # Return min and max as list
    return array([min(z),max(z)]) + (0 if len(z)==1 else array([-1e-20,1e-20]))

# Useful function for getting parent directory
def parent(path):
    '''
    Simple wrapper for getting absolute parent directory
    '''
    import os
    return os.path.abspath(os.path.join(path, os.pardir))+'/'

# Class for basic print manipulation
class print_format:
   magenta = '\033[95m'
   cyan = '\033[96m'
   darkcyan = '\033[36m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bold = '\033[1m'
   grey = gray = '\033[1;30m'
   ul = '\033[4m'
   end = '\033[0m'
   underline = '\033[4m'

# Function that uses the print_format class to make tag text for bold printing
def bold(string):
    return print_format.bold + string + print_format.end
def red(string):
    return print_format.red + string + print_format.end
def green(string):
    return print_format.green + string + print_format.end
def magenta(string):
    return print_format.magenta + string + print_format.end
def blue(string):
    return print_format.blue + string + print_format.end
def grey(string):
    return print_format.grey + string + print_format.end
def yellow(string):
    return print_format.yellow + string + print_format.end
def cyan(string):
    return print_format.cyan + string + print_format.end
def darkcyan(string):
    return print_format.darkcyan + string + print_format.end
def textul(string):
    return print_format.underline + string + print_format.end
def underline(string):
    return print_format.underline + string + print_format.end

# Function to produce array of color vectors
def rgb( N,                     #
         offset     = None,     #
         speed      = None,     #
         plot       = False,    #
         shift      = None,     #
         jet        = False,    #
         reverse    = False,    #
         weights    = None,     #
         verbose    = None ):   #

    #
    from numpy import array,pi,sin,arange,linspace,amax

    # If bad first intput, let the people know.
    if not isinstance( N, int ):
        msg = 'First input must be '+cyan('int')+'.'
        raise ValueError(msg)

    #
    if offset is None:
        offset = pi/4.0

    #
    if speed is None:
        speed = 2.0

    #
    if shift is None:
        shift = 0

    #
    if jet:
        offset = -pi/2.1
        shift = pi/2.0

    #
    if weights is None:
        t_range = linspace(1,0,N)
    else:
        if len(weights)==N:
            t_range = array(weights)
            t_range /= 1 if 0==amax(t_range) else amax(t_range)
        else:
            error('weights must be of length N','rgb')

    #
    if reverse:
        t_range = linspace(1,0,N)
    else:
        t_range = linspace(0,1,N)



    #
    r = array([ 1, 0, 0 ])
    g = array([ 0, 1, 0 ])
    b = array([ 0, 0, 1 ])

    #
    clr = []
    w = pi/2.0
    for t in t_range:

        #
        R = r*sin( w*t                + shift )
        G = g*sin( w*t*speed + offset + shift )
        B = b*sin( w*t + pi/2         + shift )

        #
        clr.append( abs(R+G+B) )

    #
    if plot:

        #
        from matplotlib import pyplot as p

        #
        fig = p.figure()
        fig.set_facecolor("white")

        #
        for k in range(N):
            p.plot( array([0,1]), (k+1.0)*array([1,1])/N, linewidth=20, color = clr[k] )

        #
        p.axis('equal')
        p.axis('off')

        #
        p.ylim([-1.0/N,1.0+1.0/N])
        p.show()

    #
    return array(clr)

#
def apolyfit(x,y,order=None,tol=1e-3):
    #
    from numpy import polyfit,poly1d,std,inf

    #
    givenorder = False if order is None else True

    #
    done = False; k = 0; ordermax = len(x)-1; oldr = inf
    while not done:

        order = k if givenorder is False else order
        fit = poly1d(polyfit(x,y,order))
        r = std( fit(x)-y ) / ( std(y) if std(y)>1e-15 else 1.0 )
        k += 1

        dr = oldr-r # ideally dr > 0

        if order==ordermax:
            done = True
        if dr <= tol:
            done = True
        if dr < 0:
            done = True

        if givenorder:
            done = True

    #
    return fit


# custome function for setting desirable ylimits
def pylim( x, y, axis='both', domain=None, symmetric=False, pad_y=0.1 ):
    '''Try to automatically determine nice xlim and ylim settings for the current axis'''
    #
    from matplotlib.pyplot import xlim, ylim
    from numpy import ones

    #
    if domain is None:
        mask = ones( x.shape, dtype=bool )
    else:
        mask = (x>=min(domain))*(x<=max(domain))

    #
    if axis == 'x' or axis == 'both':
        xlim( lim(x) )

    #
    if axis == 'y' or axis == 'both':
        limy = lim(y[mask]); dy = pad_y * ( limy[1]-limy[0] )
        if symmetric:
            ylim( [ -limy[-1]-dy , limy[-1]+dy ] )
        else:
            ylim( [ limy[0]-dy , limy[-1]+dy ] )

# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

#
# Use formula from wikipedia to calculate the harmonic
# See http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
# for more information.
def sYlm(s,l,m,theta,phi):

    #
    from numpy import pi,ones,sin,tan,exp,array,double,sqrt,zeros
    from scipy.misc import factorial,comb

    #
    if isinstance(theta,(float,int,double)): theta = [theta]
    if isinstance(phi,(float,int,double)): phi = [phi]
    theta = array(theta)
    phi = array(phi)

    #
    theta = array([ double(k) for k in theta ])
    phi = array([ double(k) for k in phi ])

    # Ensure regular output (i.e. no nans)
    theta[theta==0.0] = 1e-9

    # Name anonymous functions for cleaner syntax
    f = lambda k: double(factorial(k))
    c = lambda x: double(comb(x[0],x[1]))
    cot = lambda x: 1.0/double(tan(x))

    # Pre-allocatetion array for calculation (see usage below)
    if min(theta.shape)!=1 and min(phi.shape)!=1:
        X = ones( len(theta) )
        if theta.shape != phi.shape:
            error('Input dim error: theta and phi inputs must be same size.')
    else:
        X = ones( theta.shape )


    # Calcualte the "pre-sum" part of sYlm
    a = (-1.0)**(m)
    a = a * sqrt( f(l+m)*f(l-m)*(2.0*l+1) )
    a = a / sqrt( 4.0*pi*f(l+s)*f(l-s) )
    a = a * sin( theta/2.0 )**(2.0*l)
    A = a * X

    # Calcualte the "sum" part of sYlm
    B = zeros(theta.shape)
    for k in range(len(theta)):
        B[k] = 0
        for r in range(l-s+1):
            if (r+s-m <= l+s) and (r+s-m>=0) :
                a = c([l-s,r])*c([l+s,r+s-m])
                a = a * (-1)**(l-r-s)
                a = a * cot( theta[k]/2.0 )**(2*r+s-m)
                B[k] = B[k] + a

    # Calculate final output array
    Y = A*B*exp( 1j*m*phi )

    #
    if sum(abs(Y.imag)) == 1e-7:
        Y = Y.real

    #
    return Y


# Convert complex number to string in exponential form
def complex2str( x, precision=None, latex=False ):
    '''Convert complex number to string in exponential form '''
    # Import useful things
    from numpy import ndarray,angle,abs,pi
    # Handle optional precision input
    precision = 8 if precision is None else precision
    precision = -precision if precision<0 else precision
    # Create function to convert single number to string
    def c2s(y):

        # Check type
        if not isinstance(y,complex):
            msg = 'input must be complex number or numpy array of complex datatype'

        #
        handle_as_real = abs(y.imag) < (10**(-precision))

        if handle_as_real:
            #
            fmt = '%s1.%if'%(r'%',precision)
            ans_ = '%s' % ( fmt ) % y.real
        else:

            # Compute amplitude and phase
            amp,phase = abs(y),angle(y)
            # Write phase as positive number
            phase = phase+2*pi if phase<0 else phase
            # Create string
            fmt = '%s1.%if'%(r'%',precision)
            ans_ = '%s*%s%s%s' % (fmt, 'e^{' if latex else 'exp(' ,fmt, 'i}' if latex else 'j)') % (amp,phase)
            if latex: ans_ = ans_.replace('*',r'\,')

        return ans_

    # Create the final string representation
    if isinstance(x,(list,ndarray,tuple)):
        s = []
        for c in x:
            s += [c2s(c)]
        ans = ('\,+\,' if latex else ' + ').join(s)
    else:
        ans = c2s(x)
    # Return the answer
    return ans

# Calculate teh positive definite represenation of the input's complex phase
def anglep(x):
    '''Calculate teh positive definite represenation of the input's complex phase '''
    from numpy import angle,amin,pi,exp,amax
    #
    initial_shape = x.shape
    x_ = x.reshape( (x.size,) )
    #
    x_phase = angle(x_)
    C = 2*pi # max( abs(amin(x_phase)), abs(amax(x_phase))  )
    x_phase -= C
    for k,y in enumerate(x_phase):
        while y < 0:
            y += 2*pi
        x_phase[k] = y
    return x_phase.reshape(initial_shape)+C


# Sort an array, unwrap it, and then reimpose its original order
def sunwrap( a ):
    ''' Sort an array, unwrap it, and then reimpose its original order '''

    # Import useful things
    from numpy import unwrap,array,pi,amin,amax,isnan,nan,isinf,isfinite,mean

    # Flatten array by size
    true_shape = a.shape
    b = a.reshape( (a.size,) )

    # Handle non finites
    nanmap = isnan(b) | isinf(b)
    b[nanmap] = -200*pi*abs(amax(b[isfinite(b)]))

    # Sort
    chart = sorted(  range(len(b))  ,key=lambda c: b[c])

    # Apply the sort
    c = b[ chart ]

    # Unwrap the sorted
    d = unwrap(c)
    d -= 2*pi*( 1 + int(abs(amax(d))) )
    while amax(d)<0:
        d += 2*pi

    # Re-order
    rechart = sorted(  range(len(d))  ,key=lambda r: chart[r])

    # Restore non-finites
    e = d[ rechart ]
    e[nanmap] = nan

    #
    f = e - mean(e)
    pm = mean( f[f>=0] )
    mm = mean( f[f<0] )
    while pm-mm > pi:
        f[ f<0 ] += 2*pi
        mm = mean( f[f<0] )
    f += mean(e)


    # Restore true shape and return
    return f.reshape( true_shape )

#
def sunwrap_dev(X_,Y_,Z_):
    '''Given x,y,z unwrap z using x and y as coordinates'''

    #
    from numpy import unwrap,array,pi,amin,amax,isnan,nan
    from numpy import sqrt,isinf,isfinite,inf
    from numpy.linalg import norm

    #
    true_shape = X_.shape
    X = X_.reshape( (X_.size,) )
    Y = Y_.reshape( (Y_.size,) )
    Z = Z_.reshape( (Z_.size,) )

    #
    threshold = pi

    #
    skip_dex = []
    for k,z in enumerate(Z):
        #
        if isfinite(z) and ( k not in skip_dex ):
            #
            x,y = X[k],Y[k]
            #
            min_dr,z_min,j_min = inf,None,None
            for j,zp in enumerate(Z):
                if j>k:
                    dr = norm( [ X[j]-x, Y[j]-y ] )
                    if dr < min_dr:
                        min_dr = dr
                        j_min = j
                        z_min = zp
            #
            if z_min is not None:
                skip_dex.append( j_min )
                dz = z - z_min
                if dz < threshold:
                    Z[k] += 2*pi
                elif dz> threshold:
                    Z[k] -= 2*pi

    #
    ans = Z.reshape( true_shape )

    #
    return ans


# Useful identity function of two inputs --- this is here becuase pickle cannot store lambdas in python < 3
def IXY(x,y): return y

# Rudimentary single point outlier detection based on cross validation of statistical moments
# NOTE that this method is to be used sparingly. It was developed to help extrapolate NR data ti infinity
def single_outsider( A ):
    '''Rudimentary outlier detection based on cross validation of statistical moments'''

    # Import useful things
    from numpy import std,array,argmin,ones,mean

    #
    true_shape = A.shape

    #
    a = array( abs( A.reshape( (A.size,) ) ) )
    a = a - mean(a)

    #
    std_list = []
    for k in range( len(a) ):

        #
        b = [ v for v in a if v!=a[k]  ]
        std_list.append( std(b) )

    #
    std_arr = array(std_list)

    #
    s = argmin( std_arr )

    # The OUTSIDER is the data point that, when taken away, minimizes the standard deviation of the population.
    # In other words, the outsider is the point that adds the most diversity.

    mask = ones( a.shape, dtype=bool )
    mask[s] = False
    mask = mask.reshape( true_shape )

    # Return the outsider's location and a mask to help locate it within related data
    return s,mask


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Given a 1D array, determine the set of N lines that are optimally representative  #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Hey, here's a function that approximates any 1d curve as a series of lines
def romline(  domain,           # Domain of Map
              range_,           # Range of Map
              N,                # Number of Lines to keep for final linear interpolator
              positive=True,   # Toggle to use positive greedy algorithm ( where rom points are added rather than removed )
              verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/( R1 if abs(R1)!=0 else 1 )

    #
    if not positive:
        #
        done = False
        space = range( len(d) )
        raw_space = range( len(d) )
        err = lambda x: mean( abs(x) ) # std(x) #
        raw_mask = []
        while not done:
            #
            min_sigma = inf
            for k in range(len(space)):
                # Remove a trial domain point
                trial_space = list(space)
                trial_space.pop(k)
                # Determine the residual error incured by removing this trial point after linear interpolation
                # Apply linear interpolation ON the new domain TO the original domain
                trial_domain = d[ trial_space ]
                trial_range = r[ trial_space ]
                # Calculate the ROM's representation error using ONLY the points that differ from the raw domain, as all other points are perfectly represented by construction. NOTE that doing this significantly speeds up the algorithm.
                trial_mask = list( raw_mask ).append( k )
                sigma = err( linterp( trial_domain, trial_range )( d[trial_mask] ) - r[trial_mask] ) / ( err(r[trial_mask]) if err(r[trial_mask])!=0 else 1e-8  )
                #
                if sigma < min_sigma:
                    min_k = k
                    min_sigma = sigma
                    min_space = array( trial_space )

            #
            raw_mask.append( min_k )
            #
            space = list(min_space)

            #
            done = len(space) == N

        #
        rom = linterp( d[min_space], R[min_space] )
        knots = min_space

    else:
        from numpy import inf,argmin,argmax
        seed_list = [ 0, argmax(R), argmin(R), len(R)-1 ]
        min_sigma = inf
        for k in seed_list:
            trial_knots,trial_rom,trial_sigma = positive_romline( d, R, N, seed = k )
            # print trial_sigma
            if trial_sigma < min_sigma:
                knots,rom,min_sigma = trial_knots,trial_rom,trial_sigma

    #
    # print min_sigma

    return knots,rom


# Hey, here's a function related to romline
def positive_romline(   domain,           # Domain of Map
                        range_,           # Range of Map
                        N,                # Number of Lines to keep for final linear interpolator
                        seed = None,      # First point in domain (index) to use
                        verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

    #
    if seed is None:
        seed = argmax(r)
    else:
        if not isinstance(seed,int):
            msg = 'seed input must be int'
            error( msg, 'positive_romline' )

    #
    done = False
    space = [ seed ]
    domain_space = range(len(d))
    err = lambda x: mean( abs(x) ) # std(x) #
    min_space = list(space)
    while not done:
        #
        min_sigma = inf
        for k in [ a for a in domain_space if not (a in space) ]:
            # Add a trial point
            trial_space = list(space)
            trial_space.append(k)
            trial_space.sort()
            # Apply linear interpolation ON the new domain TO the original domain
            trial_domain = d[ trial_space ]
            trial_range = r[ trial_space ]
            #
            sigma = err( weights * (linterp( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
            #
            if sigma < min_sigma:
                min_k = k
                min_sigma = sigma
                min_space = array( trial_space )

        #
        space = list(min_space)
        #
        done = len(space) == N

    #
    rom = linterp( d[min_space], R[min_space] )
    knots = min_space

    return knots,rom,min_sigma



# Plot 2d surface and related scatter points
def splot(domain,scalar_range,domain2=None,scalar_range2=None,kind=None,ms=35,cbfs=12):
    '''Plot 2d surface and related scatter points '''

    # Import usefult things
    from matplotlib.pyplot import figure,plot,scatter,xlabel,ylabel,savefig,imshow,colorbar,gca
    from numpy import linspace,meshgrid,array,angle,unwrap
    from matplotlib import cm

    #
    kind = 'amp' if kind is None else kind

    #
    plot_scatter = (domain2 is not None) and (scalar_range2 is not None)

    #
    fig = figure( figsize=2*array([4,2.8]) )
    clrmap = cm.coolwarm

    #
    # Z = abs(SR) if kind=='amp' else angle(SR)
    Z = abs(scalar_range) if kind=='amp' else sunwrap(angle(scalar_range))

    #
    norm = cm.colors.Normalize(vmax=1.1*Z.max(), vmin=Z.min())

    # Plot scatter of second dataset
    if plot_scatter:
        # Set marker size
        mkr_size = ms
        # Scatter the outline of domain points
        scatter( domain2[:,0], domain2[:,1], mkr_size+5, color='k', alpha=0.6, marker='o', facecolors='none' )
        # Scatter the location of domain points and color by value
        Z_ = abs(scalar_range2) if kind=='amp' else sunwrap(angle(scalar_range2))
        scatter( domain2[:,0],domain2[:,1], mkr_size, c=Z_,
                 marker='o',
                 cmap=clrmap, norm=norm, edgecolors='none' )

    #
    extent = (domain[:,0].min(),domain[:,0].max(),domain[:,1].min(),domain[:,1].max())
    im = imshow(Z, extent=extent, aspect='auto',
                    cmap=clrmap, origin='lower', norm=norm )

    #
    cb = colorbar()
    cb_range = linspace(Z.min(),Z.max(),5)
    cb.set_ticks( cb_range )
    cb.set_ticklabels( [ '%1.3f'%k for k in cb_range ] )
    cb.ax.tick_params(labelsize=cbfs)

    #
    return gca()
