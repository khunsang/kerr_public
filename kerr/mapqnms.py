

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
'''Class for boxes in complex frequency space'''
# The routines of this class assist in the solving and classification of
# QNM solutions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
class cwbox:
    # ************************************************************* #
    # This is a class to fascilitate the solving of leaver's equations varying
    # the real and comlex frequency components, and optimizing over the separation constants.
    # ************************************************************* #
    def __init__(this,
                 l,m,               # QNM indeces
                 cwr,               # Center coordinate of real part
                 cwc,               # Center coordinate of imag part
                 wid,               # box width
                 hig,               # box height
                 res = 50,          # Number of gridpoints in each dimension
                 parent = None,     # Parent of current object
                 sc = None,         # optiional holder for separatino constant
                 verbose = False,   # be verbose
                 maxn = None,       # Overtones with n>maxn will be actively ignored. NOTE that by convention n>=0.
                 smallboxes = True, # Toggle for using small boxes for new solutions
                 **kwargs ):
        #
        from numpy import array,complex128,meshgrid,float128
        #
        this.verbose,this.res = verbose,res
        # Store QNM ideces
        this.l,this.m = l,m
        # Set box params
        this.width,this.height = None,None
        this.setboxprops(cwr,cwc,wid,hig,res,sc=sc)
        # Initial a list of children: if a box contains multiple solutions, then it is split according to each solutions location
        this.children = [this]
        # Point the object to its parent
        this.parent = parent
        #
        this.__jf__ = []
        # temp grid of separation constants
        this.__scgrid__ = []
        # current value of scalarized work-function
        this.__lvrfmin__ = None
        # Dictionary for high-level data: the data of all of this object's children is collected here
        this.data = {}
        this.dataformat = '{ ... (l,m,n,tail_flag) : { "jf":[...],"cw":[...],"sc":[...],"lvrfmin":[...] } ... }'
        # Dictionary for low-level data: If this object is fundamental, then its data will be stored here in the same format as above
        this.__data__ = {}
        # QNM label: (l,m,n,t), NOTE that "t" is 0 if the QNM is not a power-law tail and 1 otherwise
        this.__label__ = ()
        # Counter for the number of times map hass benn called on this object
        this.mapcount = 0
        # Default value for temporary separation constant
        this.__sc__ = 4.0
        # Maximum overtone label allowed.  NOTE that by convention n>=0.
        this.__maxn__ = maxn
        #
        this.__removeme__ = False
        #
        this.__smallboxes__ = smallboxes

    #################################################################
    '''************************************************************ #
              Set box params & separation constant center
    # ************************************************************'''
    #################################################################
    def setboxprops(this,cwr,cwc,wid,hig,res,sc=None,data=None,pec=None):
        # import maths and other
        from numpy import complex128,float128,array,linspace
        import matplotlib.patches as patches
        # set props for box geometry
        this.center = array([cwr,cwc])
        this.__cw__ = cwr + 1j*cwc          # Store cw for convinience

        # Boxes may only shrink. NOTE that this is usefull as some poetntial solutions, or unwanted solutions may be reomved, and we want to avoid finding them again. NOTE that this would be nice to implement, but it currently brakes the root finding.
        this.width,this.height  = float128( abs(wid) ),float128( abs(hig) )
        # if (this.width is None) or (this.height is None):
        #     this.width,this.height  = float128( abs(wid) ),float128( abs(hig) )
        # else:
        #     this.width,this.height  = min(float128( abs(wid) ),this.width),min(this.height,float128( abs(hig) ))

        this.limit  = array([this.center[0]-this.width/2.0,       # real min
                             this.center[0]+this.width/2.0,       # real max
                             this.center[1]-this.height/2.0,      # imag min
                             this.center[1]+this.height/2.0])     # imag max
        this.wr_range = linspace( this.limit[0], this.limit[1], res )
        this.wc_range = linspace( this.limit[2], this.limit[3], res )
        # Set patch object for plotting. NOTE the negative sign exists here per convention
        if None is pec: pec = 'k'
        this.patch = patches.Rectangle( (min(this.limit[0:2]), min(-this.limit[2:4]) ), this.width, this.height, fill=False, edgecolor=pec, alpha=0.4, linestyle='dotted' )
        # set holder for separation constant value
        if sc is not None:
            this.__sc__ = sc
        # Initiate the data holder for this box. The data holder will contain lists of spin, official cw and sc values
        if data is not None:
            this.data=data

    #################################################################
    '''************************************************************ #
                Map the potential solutions in this box
    # ************************************************************'''
    #################################################################
    def map(this,jf):

        # Import useful things
        from kerr import localmins # finds local minima of a 2D array
        from kerr.basics import alert,green,yellow,cyan,bold,magenta,blue
        from numpy import array,delete,ones

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        # Add the input jf to the list of jf values. NOTE that this is not the primary recommended list for referencing jf. Please use the "data" field instead.
        this.__jf__.append(jf)
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

        #

        if this.verbose:
            if this.parent is None:
                alert('\n\n# '+'--'*40+' #\n'+blue(bold('Attempting to map qnm solutions for: jf = %1.8f'%(jf)))+'\n# '+'--'*40+' #\n','map')
            else:
                print '\n# '+'..'*40+' #\n'+blue('jf = %1.8f,  label = %s'%(jf,this.__label__))+'\n# '+'..'*40+' #'

        # Map solutions using discrete grid
        if this.isfundamental():
            # Brute-force calculate solutions to leaver's equations
            if this.verbose: alert('Solvinq Leaver''s Eqns over grid','map')
            this.__x__,this.__scgrid__ = this.lvrgridsolve(jf)
            # Use a local-min finder to estimate the qnm locations for the grid of work function values, x
            if this.verbose: alert('Searching for local minima. Ignoring mins on boundaries.','map')
            this.__localmin__ = localmins(this.__x__,edge_ignore=True)
            if this.verbose: alert('Number of local minima found: %s.'%magenta('%i'%(len(array(this.__localmin__)[0]))),'map')
            # If needed, split the box into sub-boxes: Give the current box children!
            this.splitcenter() # NOTE that if there is only one lcal min, then no split takes place
            # So far QNM solutions have been estimates mthat have discretization error. Now, we wish to refine the
            # solutions using optimization.
            if this.verbose: alert('Refining QNM solution locations using a hybrid strategy.','map')
            this.refine(jf)
        else:
            # Map solutions for all children
            for child in [ k for k in this.children if this is not k ]:
                child.map(jf)

        # Collect QNM solution data for this BH spin. NOTE that only non-fundamental objects are curated
        if this.verbose: alert('Collecting final QNM solution information ...','map')
        this.curate(jf)

        # Remove duplicate solutions
        this.validatechildren()

        #
        if this.verbose: alert('Mapping of Kerr QNM with (l,m)=(%i,%i) within box now complete for this box.' % (this.l,this.m ) ,'map')

        # Some book-keeping on the number of times this object has been mapped
        this.mapcount += 1

    # For the given bh spin, collect all QNM frequencies and separation constants within the current box
    # NOTE that the outputs are coincident lists
    def curate(this,jf):

        #
        from numpy import arange,array,sign

        #
        children = this.collectchildren()
        cwlist,sclist = [ child.__cw__ for child in children ],[ child.__sc__ for child in children ]
        if this.isfundamental():
            cwlist.append( this.__cw__ )
            sclist.append( this.__sc__ )

        # sort the output lists by the imaginary part of the cw values
        sbn = lambda k: abs( cwlist[k].imag ) # Sort By Overtone(N)
        space = arange( len(cwlist) )
        map_  = sorted( space, key=sbn )
        std_cwlist = array( [ cwlist[k] for k in map_ ] )
        std_sclist = array( [ sclist[k] for k in map_ ] )

        # ---------------------------------------------------------- #
        # Separate positive, zero and negative frequency solutions
        # ---------------------------------------------------------- #

        # Solutions with frequencies less than this value will be considered to be power-laws
        pltol = 0.01
        # Frequencies
        sorted_cw_pos = list(  std_cwlist[ (sign(std_cwlist.real) == sign(this.m)) * (abs(std_cwlist.real)>pltol) ]  )
        sorted_cw_neg = list(  std_cwlist[ (sign(std_cwlist.real) ==-sign(this.m)) * (abs(std_cwlist.real)>pltol) ]  )
        sorted_cw_zro = list(  std_cwlist[ abs(std_cwlist.real)<=pltol ]  )

        # Create a dictionary between (cw,sc) and child objects
        A,B = {},{}
        for child in children:
            A[child] = ( child.__cw__, child.__sc__ )
            B[ A[child] ] = child
        #
        def inferlabel( cwsc ):
            cw,sc = cwsc[0],cwsc[1]
            ll = this.l
            if abs(cw.real)<pltol :
                # power-law decay
                tt = 1
                nn = sorted_cw_zro.index( cw )
                mm = this.m
            else:
                tt = 0
                if sign(this.m)==sign(cw.real):
                    # prograde
                    mm = this.m
                    nn = sorted_cw_pos.index( cw )
                else:
                    # retrograde
                    mm = -1 * this.m
                    nn = sorted_cw_neg.index( cw )
            #
            return (ll,mm,nn,tt)

        # ---------------------------------------------------------- #
        # Create a dictionary to keep track of potential solutions
        # ---------------------------------------------------------- #
        label = {}
        for child in children:
            cwsc = ( child.__cw__, child.__sc__ )
            label[child] = inferlabel( cwsc )
            child.__label__ = label[child]

        #
        this.label = label

        '''
        IMPORTANT: Here it is assumed that the solutions will change in a continuous manner, and that after the first mapping, no new solutions are of interest, unless a box-split occurs.
        '''

        # Store the high-level data product
        for child in children:
            L = this.label[child]
            if not L in this.data:
                this.data[ L ] = {}
                this.data[ L ][ 'jf' ] = [jf]
                this.data[ L ][ 'cw' ] = [ child.__cw__ ]
                this.data[ L ][ 'sc' ] = [ child.__sc__ ]
                this.data[ L ][ 'lvrfmin' ] = [ child.__lvrfmin__ ]
            else:
                this.data[ L ][ 'jf' ].append(jf)
                this.data[ L ][ 'cw' ].append(child.__cw__)
                this.data[ L ][ 'sc' ].append(child.__sc__)
                this.data[ L ][ 'lvrfmin' ].append(child.__lvrfmin__)
            # Store the information to this child also
            child.__data__['jf'] = this.data[ L ][ 'jf' ]
            child.__data__['cw'] = this.data[ L ][ 'cw' ]
            child.__data__['sc'] = this.data[ L ][ 'sc' ]
            child.__data__['lvrfmin'] = this.data[ L ][ 'lvrfmin' ]

    # Refine the box center using fminsearch
    def refine(this,jf):
        # Import useful things
        from numpy import complex128,array,linalg,log,exp,abs
        from scipy.optimize import fmin,root,fmin_tnc,fmin_slsqp
        from kerr.pttools import leaver_workfunction,scberti
        from kerr.basics import alert,say,magenta,bold,green,cyan,yellow
        from kerr import localmins # finds local minima of a 2D array

        #
        if this.isfundamental():
            # use the box center for refined minimization
            CW = complex128( this.center[0] + 1j*this.center[1] )
            # SC = this.__sc__
            SC = scberti( CW*jf, this.l, this.m )
            state = [ CW.real,CW.imag, SC.real,SC.imag ]

            #
            retrycount,maxretrycount,done = -1,1,False
            while done is False:

                #
                retrycount += 1

                #
                if retrycount==0:
                    alert(cyan('* Constructing guess using scberti-grid or extrap.'),'refine')
                    state = this.guess(jf,gridguess=state)
                else:
                    alert(cyan('* Constructing guess using 4D-grid or extrap.'),'refine')
                    state = this.guess(jf)

                # Solve leaver's equations using a hybrid strategy
                cw,sc,this.__lvrfmin__,retry = this.lvrsolve(jf,state)

                # If the root finder had some trouble, then mark this box with a warning (for plotting)
                done = (not retry) or (retrycount>=maxretrycount)
                #
                if retry:

                    newres = 2*this.res

                    if this.verbose:
                        msg = yellow( 'The current function value is %s. Retrying root finding for %ind time with higher resolution pre-grid, and brute-force 4D.'%(this.__lvrfmin__, retrycount+2) )
                        alert(msg,'refine')
                        # say('Retrying.','refine')

                    # Increase the resolution of the box
                    this.setboxprops(this.__cw__.real,this.__cw__.imag,this.width,this.height,newres,sc=this.__sc__)
                    # NOTE that the commented out code below is depreciated by the use of guess() above.
                    # # Brute force solve again
                    # this.__x__,this.__scgrid__ = this.lvrgridsolve(jf,fullopt=True)
                    # # Use the first local min as a guess
                    # this.__localmin__ = localmins(this.__x__,edge_ignore=True)
                    # state = this.grids2states()[0]

            # if this.verbose: print X.message+' The final function value is %s'%(this.__lvrfmin__)
            if this.verbose: print 'The final function value is '+green(bold('%s'%(this.__lvrfmin__)))

            if this.verbose:
                print '\n\t Geuss   cw: %s' % CW
                print '\t Optimal cw: %s' % cw
                print '\t Approx  sc: %s' % scberti( CW*jf, this.l, this.m )
                print '\t Geuss   sc: %s' % (state[2]+1j*state[3])
                print '\t Optimal sc: %s\n' % sc

            # Set the core properties of the new box
            this.setboxprops( cw.real, cw.imag, this.width,this.height,this.res,sc=sc )

            # Rescale this object's boxes based on new centers
            this.parent.sensescale()

        else:
            #
            for child in [ k for k in this.children if this is not k ]:
                child.refine(jf)

    # Determine if the current object has more than itself as a child
    def isfundamental(this):
        return len(this.children) is 1

    # ************************************************************* #
    # Determin whether to split this box into sub-boxes (i.e. children)
    # and if needed, split
    # ************************************************************* #
    def splitcenter(this):
        from numpy import array,zeros,linalg,inf,mean,amax,amin,sqrt
        from kerr.basics import magenta,bold,alert,error,red,warning,yellow
        mins =  this.__localmin__
        num_solutions = len(array(mins)[0])
        if num_solutions > 1: # Split the box
            # for each min
            for k in range(len(mins[0])):

                # construct the center location
                kr = mins[1][k]; wr = this.wr_range[ kr ]
                kc = mins[0][k]; wc = this.wc_range[ kc ]
                sc = this.__scgrid__[kr,kc]
                # Determine the resolution of the new box
                res = int( max( 20, 1.5*float(this.res)/num_solutions ) )
                # Create the new child. NOTE that the child's dimensions will be set below using a standard method.
                child = cwbox( this.l,this.m,wr,wc,0,0, res, parent=this, sc=sc, verbose=this.verbose )
                # Add the new box to the current box's child list
                this.children.append( child )

            # NOTE that here we set the box dimensions of all children using the relative distances between them
            this.sensescale()

            # Now redefine the box size to contain all children
            # NOTE that this step exists only to ensure that the box always contains all of its children's centers
            children = this.collectchildren()
            wr = array( [ child.center[0] for child in children ] )
            wc = array( [ child.center[1] for child in children ] )
            width = amax(wr)-amin(wr)
            height = amax(wc)-amin(wc)
            cwr = mean(wr)
            cwc = mean(wc)
            this.setboxprops( cwr,cwc,width,height,this.res,sc=sc )

        elif num_solutions == 1:
            # construcut the center location
            k = 0 # there should be only one local min
            kr = mins[1][k]
            kc = mins[0][k]
            wr = this.wr_range[ kr ]
            wc = this.wc_range[ kc ]
            # retrieve associated separation constant
            sc  = this.__scgrid__[kr,kc]
            # Recenter the box on the current min
            this.setboxprops(wr,wc,this.width,this.height,this.res,sc=sc)
        else:
            #
            if len(this.__jf__)>3:
                alert('Invalid number of local minima found: %s.'% (magenta(bold('%s'%num_solutions))), 'splitcenter' )
                # Use the extrapolated values as a guess?
                alert(yellow('Now trying to use extrapolation, wrather than grid guess, to center the current box.'),'splitcenter')
                #
                guess = this.guess(this.__jf__[-1],gridguess=[1.0,1.0,4.0,1.0])
                wr,wc,cr,cc = guess[0],guess[1],guess[2],guess[3]
                sc = cr+1j*cc
                # Recenter the box on the current min
                this.setboxprops(wr,wc,this.width,this.height,this.res,sc=sc)
            else:
                warning('Invalid number of local minima found: %s. This box will be removed. NOTE that this may not be what you want, and further inspection may be warranted.'% (magenta(bold('%s'%num_solutions))), 'splitcenter' )
                this.__removeme__ = True



    # Validate children: Remove duplicates
    def validatechildren(this):
        #
        from numpy import linalg,array
        from kerr import alert,yellow,cyan,blue,magenta
        tol = 1e-5

        #
        if not this.isfundamental():

            #
            children = this.collectchildren()
            initial_count = len(children)

            # Remove identical twins
            for a,tom in enumerate( children ):
                for b,tim in enumerate( children ):
                    if b>a:
                        if linalg.norm(array(tom.center)-array(tim.center)) < tol:
                            tim.parent.children.remove(tim)
                            del tim
                            break

            # Remove overtones over the max label
            if this.__maxn__ is not None:
                for k,child in enumerate(this.collectchildren()):
                    if child.__label__[2] > this.__maxn__:
                        if this.verbose:
                            msg = 'Removing overtone '+yellow('%s'%list(child.__label__))+' becuase its label is higher than the allowed value specified.'
                            alert(msg,'validatechildren')
                        this.label.pop( child.__label__ , None)
                        child.parent.children.remove(child)
                        del child

            # Remove all boxes marked for deletion
            for child in this.collectchildren():
                if child.__removeme__:
                    this.label.pop( child.__label__, None )
                    child.parent.children.remove( child )
                    del child

            #
            final_count = len( this.collectchildren() )
            #
            if this.verbose:
                if final_count != initial_count:
                    alert( yellow('%i children have been removed, and %i remain.') % (-final_count+initial_count,final_count) ,'validatechildren')
                else:
                    alert( 'All children have been deemed valid.', 'validatechildren' )

    # Method for collecting all fundamental children
    def collectchildren(this,children=None):
        #
        if children is None:
            children = []
        #
        if this.isfundamental():
            children.append(this)
        else:
            for child in [ k for k in this.children if k is not this ]:
                children += child.collectchildren()
        #
        return children

    # Method to plot solutions
    def plot(this,fig=None,show=False,showlabel=False):
        #
        from numpy import array,amin,amax,sign
        from matplotlib.pyplot import plot,xlim,ylim,xlabel,ylabel,title,figure,gca,text
        from matplotlib.pyplot import show as show_

        #
        children = this.collectchildren()
        wr = array( [ child.center[0] for child in children ] )
        wc =-array( [ child.center[1] for child in children ] )
        wr_min,wr_max = amin(wr),amax(wr)
        wc_min,wc_max = amin(wc),amax(wc)

        padscale = 0.15
        padr,padc = 1.5*padscale*(wr_max-wr_min), padscale*(wc_max-wc_min)
        wr_min -= padr; wr_max += padr
        wc_min -= padc; wc_max += padc
        #
        if fig is None:
            # fig = figure( figsize=12*array((wr_max-wr_min, wc_max-wc_min))/(wr_max-wr_min), dpi=200, facecolor='w', edgecolor='k' )
            fig = figure( figsize=12.0*array((4.5, 3))/4.0, dpi=200, facecolor='w', edgecolor='k' )
        #
        xlim( [wr_min,wr_max] )
        ylim( [wc_min,wc_max] )
        ax = gca()
        #
        for child in children:
            plot( child.center[0],-child.center[1], '+k', ms=10 )
            ax.add_patch( child.patch )
            if showlabel:
                text( child.center[0]+sign(child.center[0])*child.width/2,-(child.center[1]+child.height/2),
                      '$(%i,%i,%i,%i)$'%(this.label[child]),
                      ha=('right' if sign(child.center[0])<0 else 'left' ),
                      fontsize=10,
                      alpha=0.9 )
        #
        xlabel(r'$\mathrm{re}\;\tilde\omega_{%i%i}$'%(this.l,this.m))
        ylabel(r'-$\mathrm{im}\;\tilde\omega_{%i%i}$'%(this.l,this.m))
        title(r'$j_f = %1.6f$'%this.__jf__[-1],fontsize=18)
        #
        if show: show_()

    # ************************************************************* #
    # Solve leaver's equations in a given box=[wr_range,wc_range]
    # NOTE that the box is a list, not an array
    # ************************************************************* #
    def lvrgridsolve(this,jf=0,fullopt=False):
        # Import maths
        from numpy import linalg,complex128,ones,array
        from kerr.pttools import scberti
        from kerr.pttools import leaver_workfunction
        from scipy.optimize import fmin,root
        import sys

        # Pre-allocate an array that will hold work function values
        x = ones(  ( this.wc_range.size,this.wr_range.size )  )
        # Pre-allocate an array that will hold sep const vals
        scgrid = ones(  ( this.wc_range.size,this.wc_range.size ), dtype=complex128  )
        # Solve over the grid
        for i,wr in enumerate( this.wr_range ):
            for j,wc in enumerate( this.wc_range ):
                # Costruct the complex frequency for this i and j
                cw = complex128( wr+1j*wc )

                # # Define the intermediate work function to be used for this iteration
                # fun = lambda SC: linalg.norm( array(leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]] )) )
                # # For this complex frequency, optimize over separation constant using initial guess
                # SC0_= scberti( cw*jf, this.l, this.m ) # Use Berti's analytic prediction as a guess
                # SC0 = [SC0_.real,SC0_.imag]
                # X  = fmin( fun, SC0, disp=False, full_output=True, maxiter=1 )
                # # Store work function value
                # x[j][i] = X[1]
                # # Store sep const vals
                # scgrid[j][i] = X[0][0] + 1j*X[0][1]

                if fullopt is False:

                    # Define the intermediate work function to be used for this iteration
                    fun = lambda SC: linalg.norm( array(leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]] )) )
                    # For this complex frequency, optimize over separation constant using initial guess
                    SC0_= scberti( cw*jf, this.l, this.m ) # Use Berti's analytic prediction as a guess
                    SC0 = [SC0_.real,SC0_.imag]
                    # Store work function value
                    x[j][i] = fun(SC0)
                    # Store sep const vals
                    scgrid[j][i] = SC0_

                else:

                    SC0_= scberti( cw*jf, this.l, this.m ) # Use Berti's analytic prediction as a guess
                    SC0 = [SC0_.real,SC0_.imag,0,0]
                    #cfun = lambda Y: [ Y[0]+abs(Y[3]), Y[1]+abs(Y[2]) ]
                    fun = lambda SC:leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]] )
                    X  = root( fun, SC0 )
                    scgrid[j][i] = X.x[0]+1j*X.x[1]
                    x[j][i] = linalg.norm( array(X.fun) )


            if this.verbose:
                sys.stdout.flush()
                print '.',

        if this.verbose: print 'Done.'
        # return work function values AND the optimal separation constants
        return x,scgrid

    # Convert output of localmin to a state vector for minimization
    def grids2states(this):

        #
        from numpy import complex128
        state = []

        #
        for k in range( len(this.__localmin__[0]) ):
            #
            kr,kc = this.__localmin__[1][k], this.__localmin__[0][k]
            cw = complex128( this.wr_range[kr] + 1j*this.wc_range[kc] )
            sc = complex128( this.__scgrid__[kr,kc] )
            #
            state.append( [cw.real,cw.imag,sc.real,sc.imag] )

        #
        return state

    # Get guess either from local min, or from extrapolation of past data
    def guess(this,jf,gridguess=None):
        #
        from kerr.pttools import leaver_workfunction
        from kerr.basics import alert,magenta,apolyfit
        from kerr import localmins
        from numpy import array,linalg,arange,complex128,allclose,nan
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        # Get a guess from the localmin
        if gridguess is None:
            this.__x__,this.__scgrid__ = this.lvrgridsolve(jf,fullopt=True)
            this.__localmin__ = localmins(this.__x__,edge_ignore=True)
            guess1 = this.grids2states()[0]
        else:
            guess1 = gridguess
        # Get a guess from extrapolation ( performed in curate() )
        guess2 = [ v for v in guess1 ]
        if this.mapcount > 3:
            # if there are three map points, try to use polynomial fitting to determine the state at the current jf value
            nn = len(this.__data__['jf'])
            order = min(2,nn)
            #
            xx = array(this.__data__['jf'])[-4:]
            #
            yy = array(this.__data__['cw'])[-4:]
            yr = apolyfit( xx, yy.real, order )(jf)
            yc = apolyfit( yy.real, yy.imag, order )(yr)
            cw = complex128( yr + 1j*yc )
            #
            zz = array(this.__data__['sc'])[-4:]
            zr = apolyfit( xx, zz.real, order  )(jf)
            zc = apolyfit( zz.real, zz.imag, order  )(zr)
            sc = complex128( zr + 1j*zc )
            #
            guess2 = [ cw.real, cw.imag, sc.real, sc.imag ]
        # Determine the best guess
        if not ( allclose(guess1,guess2) ):
            x1 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess1 ) )
            x2 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess2 ) )
            alert(magenta('The function value at guess from grid is:   %s'%x1),'guess')
            alert(magenta('The function value at guess from extrap is: %s'%x2),'guess')
            if x2 is nan:
                x2 = 100.0*x1
            if x1<x2:
                guess = guess1
                alert(magenta('Using the guess from the grid.'),'guess')
            else:
                guess = guess2
                alert(magenta('Using the guess from extrapolation.'),'guess')
        else:
            x1 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess1 ) )
            guess = guess1
            alert(magenta('The function value at guess from grid is %s'%x1),'guess')
        # Return the guess solution
        return guess


    # Determine whether the current box contains a complex frequency given an iterable whose first two entries are the real and imag part of the complex frequency
    def contains(this,guess):
        #
        cwrmin = min( this.limit[:2] )
        cwrmax = max( this.limit[:2] )
        cwcmin = min( this.limit[2:] )
        cwcmax = max( this.limit[2:] )
        #
        isin  = True
        isin = isin and ( guess[0]<cwrmax )
        isin = isin and ( guess[0]>cwrmin )
        isin = isin and ( guess[1]<cwcmax )
        isin = isin and ( guess[1]>cwcmin )
        #
        return isin


    # Try solving the 4D equation near a single guess value [ cw.real cw.imag sc.real sc.imag ]
    def lvrsolve(this,jf,guess,tol=1e-8):

        # Import Maths
        from numpy import log,exp,linalg,array
        from scipy.optimize import root,fmin,minimize
        from kerr.pttools import leaver_workfunction
        from kerr import alert,red

        # Try using root
        # Define the intermediate work function to be used for this iteration
        fun = lambda STATE: log( 1.0 + abs(array(leaver_workfunction( jf,this.l,this.m, STATE ))) )
        X  = root( fun, guess, tol=tol )
        cw1,sc1 = X.x[0]+1j*X.x[1], X.x[2]+1j*X.x[3]
        __lvrfmin1__ = linalg.norm(array( exp(X.fun)-1.0 ))
        retry1 = ( 'not making good progress' in X.message.lower() ) or ( 'error' in X.message.lower() )


        # Try using fmin
        # Define the intermediate work function to be used for this iteration
        fun = lambda STATE: log(linalg.norm(  leaver_workfunction( jf,this.l,this.m, STATE )  ))
        X  = fmin( fun, guess, disp=False, full_output=True, ftol=tol )
        cw2,sc2 = X[0][0]+1j*X[0][1], X[0][2]+1j*X[0][3]
        __lvrfmin2__ = exp(X[1])
        retry2 = this.__lvrfmin__ > 1e-3

        # Use the solution that converged the fastest to avoid solutions that have wandered significantly from the initial guess OR use the solution with the smallest fmin
        if __lvrfmin1__ < __lvrfmin2__ : # use the fmin value for convenience
            cw,sc,retry = cw1,sc1,retry1
            __lvrfmin__ = __lvrfmin1__
        else:
            cw,sc,retry = cw2,sc2,retry2
            __lvrfmin__ = __lvrfmin2__

        if not this.contains( [cw.real,cw.imag] ):
            alert(red('Trial solution found to be outside of box. I will now try to use a bounded solver, but the performance may be suboptimal.'),'lvrsolve')

            s = 2.0
            cwrmin = min( this.center[0]-this.width/s, this.center[0]+this.width/s )
            cwrmax = max( this.center[0]-this.width/s, this.center[0]+this.width/s )
            cwcmin = min( this.center[1]-this.height/s, this.center[1]+this.height/s )
            cwcmax = max( this.center[1]-this.height/s, this.center[1]+this.height/s )
            scrmin = min( this.__sc__.real-this.width/s, this.__sc__.real+this.width/s )
            scrmax = max( this.__sc__.real-this.width/s, this.__sc__.real+this.width/s )
            sccmin = min( this.__sc__.imag-this.height/s, this.__sc__.imag+this.height/s )
            sccmax = max( this.__sc__.imag-this.height/s, this.__sc__.imag+this.height/s )

            bounds = [ (cwrmin,cwrmax), (cwcmin,cwcmax), (scrmin,scrmax), (sccmin,sccmax) ]

            # Try using minimize
            # Define the intermediate work function to be used for this iteration
            fun = lambda STATE: log(linalg.norm(  leaver_workfunction( jf,this.l,this.m, STATE )  ))
            X  = minimize( fun, guess, options={'disp':False}, tol=tol, bounds=bounds )
            cw,sc = X.x[0]+1j*X.x[1], X.x[2]+1j*X.x[3]
            __lvrfmin__ = exp(X.fun)


        # Always retry if the solution is outside of the box
        if not this.contains( [cw.real,cw.imag] ):
            retry = True
            alert(red('Retrying because the trial solution is outside of the box.'),'lvrsolve')

        # Don't retry if fval is small
        if __lvrfmin__ > 1e-3:
            retry = True
            alert(red('Retrying because the trial fmin value is greater than 1e-3.'),'lvrsolve')

        # Don't retry if fval is small
        if retry and (__lvrfmin__ < 1e-4):
            retry = False
            alert(red('Not retrying becuase the fmin value is low.'),'lvrsolve')

        # Return the solution
        return cw,sc,__lvrfmin__,retry

    # Given a box's children, resize the boxes relative to child locations: no boxes overlap
    def sensescale(this):

        #
        from numpy import array,inf,linalg,sqrt
        from kerr import alert

        #
        children = this.collectchildren()

        # Let my people know.
        if this.verbose:
            alert('Sensing the scale of the current object\'s sub-boxes.','sensescale')

        # Determine the distance between this min, and its closest neighbor
        scalar = sqrt(2) if (not this.__smallboxes__) else 2.0*sqrt(2.0)
        for tom in children:

            d = inf
            for jerry in [ kid for kid in children if kid is not tom ]:

                r = array(tom.center)
                r_= array(jerry.center)
                d_= linalg.norm(r_-r)
                if d_ < d:
                    d = d_

            # Use the smallest distance found to determine a box size
            s = d/scalar
            width = s; height = s; res = int( max( 20, 1.5*float(this.res)/len(children) ) ) if (len(children)>1) else this.res

            # Define the new box size for this child
            tom.setboxprops( tom.center[0], tom.center[1], width, height, res )
