

'''

# NOTES
--
    * write each mdoel as a class
# Wish List
--
    * Figure out the normalization constants for spheroidal harmonics

'''


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
#%%         Class for the MMRDNP Model           %%#
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
class mmrdnp:

    '''
    MultiMode Ringdown Non-Precessing (MMRDNP)

    Differs from MMRDNS in that:
        * its amplutides are relative to peak strain
        * its ampltides are naturally scaled such that the time origin is at the peak (not true for MMRDNS)
        * the model is for pherical rather than spheroidal modes
    '''

    name = 'MMRDNP'
    date = 'January 2017'
    purpose = 'Reference for LALSimulation Implementation --> Test GR on GW150914'
    author = 'Lionel London'
    email = ['lionel.london@ligo.org','londonl@cardiff.ac.uk','pilondon2@gmail.com','london@gatech.edu']

    # String or int used to reference model iteration number
    # Thisis called "keyword" in the modelrd class
    model_id = '090117v2'

    # Mode coordinates for model: (ll mm Spherical) (l m n spheroidal) 5 total
    # jk_ = [ (2,2,2,2,0), (2,1,2,1,0),(3,2,3,2,0) ]
    jk_ = [ (2,2,2,2,0), (2,2,2,2,1),
            (2,1,2,1,0),
            (3,2,3,2,0), (3,2,2,2,0),
            (3,3,3,3,0), (3,3,3,3,1),
            (4,4,4,4,0),
            (4,3,4,3,0), (4,3,3,3,0) ]
    # Add negative m modes to model. NOTE that the QNM are simply conjugate
    jk__,llmm = list(jk_),[]
    for jk in jk__:
        ll,mm,l,m,n = jk
        jk_ += [ (ll,-mm,l,-m,n) ]
        llmm.append( (ll,mm) )
        llmm.append( (ll,-mm) )
    jk_  = sorted( list( set(jk_)  ), key=lambda x:x[0]+abs(x[1]) )
    llmm = sorted( list( set(llmm) ), key=lambda x:x[0]+abs(x[1]) )

    # NOTE that the mode amplitudes in this model were estimated differently than those for MMRDNS
    # * In MMRDNS, the fitting region started 10 M after the peak luminosity, and the region's time
    #   series was set to start at 0 just before fitting. Both choices affect the amplitude value
    # * In MMRDNP, the fitting region started 10 M after the peak strain, and the region's time
    #   series stated with 10M upon fitting.
    # The result of this difference is that the calibration T0 needs to be set to zero here.
    calibration_T0 = 0 #

    from numpy import linspace
    t = linspace(calibration_T0,20*calibration_T0,1e3)
    del linspace

    # This class is all about the methods
    def __init__(this):
        return None

    #Eval single QNM for mmrdns in the TIME DOMAIN
    @staticmethod
    def meval_mode( ll,mm,                # Spherical Harmonic indeces
                    l,m,n,                # QNM indeces
                    m1,m2,chi1,chi2,
                    params = None,        # Dictionary of optional parameters
                    verbose = False,      # let the people know
                    kind=None,            # strain or psi4
                    gwfout=False,
                    plot = False ):

        # Import usefult things
        from numpy import linspace,array,exp,log
        from kerr.formula.ksm2_cw import CW as cwfit
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            from kerr import rgb,pylim,lim
            import matplotlib as mpl
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16

        eta,chi_s = mmrdnp.eta_chi_s(m1,m2,chi1,chi2)

        # handle empty parameter input
        if params is None: params = {}
        # get the amplitude
        A = mmrdnp.Afit(ll,mm,l,m,n,eta,chi_s)
        # Get the final spin and mass
        Mf,jf = mmrdnp.Mfjf(m1,m2,chi1,chi2)
        # Get the mode frequencies, allow for input of deviations
        cw = cwfit[(l,m,n)](jf) + ( 0 if not ('dcw' in params.keys()) else params['dcw'] )
        # Scale the ferquency by the final mass to get appropriate Geometric values
        cw /= Mf

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Handle input type
        if kind is None:
            kind = 'strain'
        kind = kind.lower()
        if kind == 'psi4':
            # multiple by (1j*cw)^2 to effect two time derivatives
            A = - A * (cw*cw)
        elif kind == 'strain':
            # Do nothing
            _ = None
        else:
            msg = 'kind must be "psi4" or "strain" (case insensitive)'
            error(msg,'mmrdns.meval_mode')
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Construct the waveform as a lambda function.#
        # NOTE that the Amplitude are relative to 10M after the peak liminosity
        t_reff = 0 if not ('t_reff' in params.keys()) else params['t_reff']
        def h(t):
            h_ = A * exp( 1j * cw * (t+t_reff-mmrdnp.calibration_T0) )
            wfarr = array( [ t, h_.real, h_.imag ] ).T
            if gwfout:
                from nrutils import gwf
                ans = gwf( wfarr, kind= r'$rh_{%i%i%i%i%i}(t)/M$'%(ll,mm,l,m,n) if kind == 'strain' else r'$rM\psi_{%i%i%i%i%i}(t)$'%(ll,mm,l,m,n) )
            else:
                ans = h_
            return ans
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh_{%i%i%i}(t)/M$'%(l,m,n),fontsize=fs)
            title( r'MMRDNS, $(q,\eta,j_f,M_f) = (%1.4f,%1.4f,%1.4f,%1.4f,)$'%(mmrdns.eta2q(eta),eta,jf,Mf) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')

        # Return the ringown waveform FUNCTION for this QNM
        return h

    @staticmethod
    def meval( theta,           # Source inclination
               phi,             # Orbital phase of system at observation
               m1,m2,chi1,chi2,
               params = None,   # Bag of optional parameters
               verbose=True,    # Toggle to let the people know
               gwfout=False,    # Toggle for output of nrutils gwf object
               spheroidal=False,# Toggle to use only spheroidal modes
               kind=None,       # strain or psi4
               plot = False  ): # Plotting

        # Import needed things
        from numpy import array,linspace,zeros,sum,mean
        from kerr.pttools import slm
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            import matplotlib as mpl
            from kerr import rgb,pylim,lim
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        def h(t):

            # Allow for full spheroidal OR spherical model output
            if spheroidal:
                error('MMRDNP needs to be updated to hadnle spheroidal recomposision')
                h_ = sum( array( [ slm(jf,l,m,n,theta,phi,norm=True)*mmrdnp.meval_mode(l,m,n,m1,m2,chi1,chi2,params,verbose,kind=kind)(t) for l,m,n in lmn ] ), axis=0 )
            else:

                from nrutils import sYlm

                h_ = sum( array( [ sYlm(-2,ll,mm,theta,phi)*mmrdnp.meval_mode(ll,mm,l,m,n,m1,m2,chi1,chi2,params=params,verbose=verbose,kind=kind)(t) for ll,mm,l,m,n in mmrdnp.jk_ ] ), axis=0 )

            if gwfout:
                from nrutils import gwf
                wfarr = array( [t,h_.real,h_.imag] ).T
                h_ = gwf( wfarr, kind= r'$rh(t,\theta,\phi)/M$' if kind=='strain' else r'$r\psi_4(t,\theta,\phi)/M$'  )

            return h_
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh(t,\theta,\phi)/M$',fontsize=fs)
            # title( r'MMRDNS, $(q,\eta,j_f,M_f,\theta,\phi) = (%1.4f,%1.4f,%1.4f,%1.4f)$'%(mmrdnp.eta2q(eta),eta,theta,phi) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')


        #
        return h

    # Get DIMENSIONFUL complex frequency given a mass ratio and mode index
    @staticmethod
    def cwfit(l,m,n,m1,m2,chi1,chi2,params=None):

        # Import useful things
        from kerr.formula.ksm2_cw import CW as cwf

        # Handle params input
        if params is None: params = {}
        # Get the final spin and mass
        Mf,jf = mmrdnp.Mfjf(m1,m2,chi1,chi2)
        # Get the mode frequencies, allow for input of deviations
        cw = cwf[(l,m,n)](jf) + ( 0 if not ('dcw' in params.keys()) else params['dcw'] )
        # Scale the ferquency by the final mass to get appropriate Geometric values
        cw /= Mf

        #
        return cw

    #Define default fitting functions for the QNM Amplitudes
    # Write function in MATLAB
    @staticmethod
    def Afit(ll,mm,l,m,n,eta,chi_s):
        # Load the fit functions that have been output by MATLAB -- NOTE that there should is a review ipython notebook that checks these functions by comaring to fit data
        exec('from kerr.formula.mmrdnp_%s import A_strain_np as A'%mmrdnp.model_id)
        from numpy import exp,pi
        # It's useful to only calculate the abs of m once
        _m_ = abs(m)
        _mm_ = abs(mm)
        # Define the mode coordinate using inputs
        jk_ = ( (ll,_mm_), (l,_m_,n) )
        # If the requested QNM is modeled, then output the related strain amplitude.
        if jk_ in A:
            if mm>0: # The fit functions are for m>0 ...
                Ak = A[jk_](eta,chi_s)
            else:   # ... But m<0 is handled by symmetry.
                # NOTE it is suspected that the -1^ll is needed to account for
                # a convention used in NR codes
                # NOTE that the need for the added -1^ll may result from the phase alignment scheme used in the modelrd class
                Ak = ((-1)**ll) * A[jk_](eta,chi_s).conj()
        else:
            # Otherwise, raise an error.
            from kerr import warning
            msg = 'The requested QNM, %s, is not available within MMRDNP. Zero will be returned.'%[ll,mm,l,m,n]
            warning(msg,'mmrdns.Afit')
            Ak = 0.0
        #
        return Ak

    #Function to convert eta, chi1, chi2 to chi_s
    @staticmethod
    def chi1chi2eta2chis(chi1,chi2,eta):
        m1,m2 = mmrdnp.eta2m1m2( eta )
        chi_s = ( m1*chi1 + m2*chi2 ) / ( m1+m2 )
        return chi_s

    #Function to convert masses to symmetric mass ratio
    @staticmethod
    def m1m2q(m1,m2):
        return mmrdns.q2eta( float(m1)/m2 )

    #Function to convert eta to m1 and m2
    @staticmethod
    def eta2m1m2(eta):
        '''Function to convert eta to m1 and m2'''
        q = mmrdnp.eta2q(eta)
        m1,m2 = mmrdnp.q2m1m2(q)
        return m1,m2

    #Function to convert masses to symmetric mass ratio
    @staticmethod
    def q2m1m2(q):
        q = max(q,1.0/q)
        m2 = 1.0/(1.0+q)
        m1 = q*m2
        return m1,m2

    #Function to convert mass ratio to symmetric mass ratio
    @staticmethod
    def q2eta(q):
        q = float(q)
        return q / (1.0+2*q+q*q)

    # Function to convert eta to mass ratio
    @staticmethod
    def eta2q(eta):
        from numpy import sqrt
        b = 2.0 - 1.0/eta
        q  = (-b + sqrt( b*b - 4.0 ))/2.0
        return q

    #
    @staticmethod
    def Mfjf(m1,m2,chi1,chi2):
        from nrutils import jf14067295,Mf14067295
        # Get the final spin and mass
        Mf, jf = Mf14067295(m1,m2,chi1,chi2),jf14067295(m1,m2,chi1,chi2)
        # Return answers
        return Mf,jf

    # return eta,chi_s given m1,m2,chi1,chi2
    @staticmethod
    def eta_chi_s(m1,m2,chi1,chi2):
        eta = m1*m2/(m1+m2)
        chi_s = (m1*chi1) + m2*chi2 / ( m1+m2 )
        return eta,chi_s

    #
    @staticmethod
    def meval_spherical_mode( ll,mm,
                              m1,m2,chi1,chi2,
                              params=None,
                              plot=False,
                              gwfout=False,
                              kind=None,
                              mode=None,
                              verbose=False):
        #
        from kerr import ysprod
        from numpy import zeros,array
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            from kerr import rgb,pylim,lim
            from kerr.formula.mmrdns_Mfjf import Mf as Mffit
            import matplotlib as mpl
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16
        # handle empty parameter input
        if params is None: params = {}
        #
        lmax = max( [k[0] for k in mmrdnp.jk_] )
        l_range = range(2,lmax+1)
        #
        if mode is not None:
            mode_space = [mode]
        else:
            mode_space = mmrdnp.jk_

        # Function to calculate the strain waveform
        def h(t):
            # Pre-allocate
            ans = zeros( t.shape, dtype=complex )
            # For all modes in teh model
            for k in mode_space:
                # Extract mode coodrinates
                ll_,mm_,l,m,n = k
                # If the spherical indeces match
                if (ll,mm) == (ll_,mm_):
                    yk_t = mmrdnp.meval_mode( ll, mm, l, m,n, m1,m2,chi1,chi2, params = params, verbose = verbose, kind=kind )(t)
                    #
                    ans += yk_t
            #
            return ans

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh_{%i%i}(t)/M$ Spherical'%(ll,mm) if kind == 'strain' else r'$rM\psi_{%i%i}(t)$ Spherical'%(ll,mm),fontsize=fs)
            #title( r'MMRDNP, $(q,\eta,j_f,M_f) = (%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,)$'%(mmrdns.eta2q(eta),eta,chi_s,jffit(eta,chi_s),1-eradfit(eta)) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')


        #
        ans = h
        if gwfout:
            from nrutils import gwf
            def gwfh(t):
                h_ = h(t)
                wfarr = array([ t, h_.real, h_.imag ]).T
                return gwf( wfarr, kind=r'$rh_{%i%i}(t)/M$'%(ll,mm) if kind == 'strain' else r'$rM\psi_{%i%i}(t)$'%(ll,mm), l=ll,m=mm )
            ans = gwfh

        #
        return ans

    #
    @staticmethod
    def meval_spheroidal_mode( ll,mm,               #
                               eta,
                               params=None,
                               plot=False,
                               gwfout=False,
                               kind=None,            #
                               mode=None,
                               verbose=False):
        #
        from kerr.formula.mmrdns_Mfjf import jf as jffit
        from kerr import ysprod
        from numpy import zeros,array
        error('MMRDNP needs to be updated so that this method works')
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            from kerr import rgb,pylim,lim
            from kerr.formula.mmrdns_Mfjf import Mf as Mffit
            import matplotlib as mpl
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16
        # handle empty parameter input
        if params is None: params = {}
        #
        jf = jffit(eta)
        #
        lmax = max( [k[0] for k in mmrdns.lmn] )
        l_range = range(2,lmax+1)
        #
        if mode is not None:
            mode_space = [mode]
        else:
            mode_space = mmrdns.lmn
        #
        def h(t):
            #
            ans = zeros( t.shape, dtype=complex )
            #
            for k in mode_space:
                #
                l,m,n = k
                cjk = ysprod( jf, ll,mm, k )
                yk_t = mmrdns.meval_mode(l,m,n,eta,params=params,verbose=verbose,kind=kind)(t)
                #
                ans += cjk * yk_t
            #
            return ans

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh_{%i%i}(t)/M$ Spherical'%(ll,mm),fontsize=fs)
            title( r'MMRDNS, $(q,\eta,j_f,M_f) = (%1.4f,%1.4f,%1.4f,%1.4f,)$'%(mmrdns.eta2q(eta),eta,jf,Mffit(eta)) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')


        #
        ans = h
        if gwfout:
            from nrutils import gwf
            def gwfh(t):
                h_ = h(t)
                wfarr = array([ t, h_.real, h_.imag ]).T
                return gwf( wfarr, kind=r'$rh_{%i%i}(t)/M$'%(ll,mm) )
            ans = gwfh

        #
        return ans

    #
    @staticmethod
    def gwylm_meval(m1,m2,chi1,chi2):

        #
        from nrutils import scentry,gwylm
        from numpy import inf

        #
        e = scentry( None, None )
        e.m1,e.m2 = m1,m2
        e.S1,e.S2 = m1*m1*chi1,m2*m2*chi2

        #
        e.xf,e.mf = mmrdnp.Mfjf(m1,m2,chi1,chi2)
        e.default_extraction_par = inf
        e.default_level = None
        e.config = None
        e.setname = 'MMRDNP'
        e.label = 'MMRDNP'

        #
        def h_gwylm(t):

            #
            y = gwylm(e,load=False)
            y.__lmlist__ = mmrdnp.llmm
            y.__input_lmlist__ = mmrdnp.llmm

            #
            for ll,mm in mmrdnp.llmm:
                y.ylm.append( mmrdnp.meval_spherical_mode( ll, mm, m1, m2, chi1, chi2, kind='psi4', gwfout=True)(t) )
                y.hlm.append( mmrdnp.meval_spherical_mode( ll, mm, m1, m2, chi1, chi2, kind='strain', gwfout=True)(t) )

            #
            y.__curate__()

            #
            return y

        #
        return h_gwylm

    # Compare to NR multipole
    @staticmethod
    def compare_spherical_mode_to_nr(scentry_object,ll,mm,kind=None,T0=20):
        # Import useful things
        from nrutils import gwylm
        # Extract Ringdown
        y = gwylm( scentry_object, lm=[ll,mm] )
        k = y.ringdown(T0=T0,T1=None,use_peak_strain=True)# Evaluate MMRDNP at the parameters of the NR waveform
        #
        kind = 'strain' if kind is None else kind
        h = k.lm[ll,mm][kind]
        # Extract and Calc needed initial params for NR case
        m1,m2 = h.ref_scentry.m1,h.ref_scentry.m2
        chi1,chi2 = h.ref_scentry.S1[-1]/(m1*m1),h.ref_scentry.S2[-1]/(m2*m2)
        # Evaluate the model
        hnp = mmrdnp.meval_spherical_mode( ll,mm,m1,m2,chi1,chi2,gwfout=True,kind=kind )( h.t )
        # Phase align the model with the NR waveform
        # NOTE that gwf's slign method here is correct as we are only considering modes with the same mm=m
        hnp.align( h,method='average-phase',mask=h.t<(k.T0+10),verbose=True )

        labels = ('MMRDNP','NR')
        ax,fig = hnp.plot(domain='time',ref_gwf=h,labels=labels)
        ax[0].set_yscale('log',clip=True)
        ax[0].set_ylim( [ h.amp.min(), 1.2*h.amp.max() ] )
        ax[0].set_title( '$(l,m) = (%i,%i)$'%(ll,mm) )
        ax,fig = hnp.plot(domain='freq',ref_gwf=h,labels=labels)


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
#%%         Class for the MMRDNS Model           %%#
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
class mmrdns:

    name = 'MMRDNS'
    date = 'September 2016'
    purpose = 'Reference for LALSimulation Implementation --> Test GR on GW150914'
    author = 'Lionel London'
    email = ['lionel.london@ligo.org','londonl@cardiff.ac.uk','pilondon2@gmail.com','london@gatech.edu']
    lmn = [ (2,2,0),
           (2,2,1),
           (2,1,0),
           (3,3,0),
           (3,2,0),
           (4,4,0),
           (4,3,0),
           (5,5,0) ]
    calibration_T0 = 10 # The time relative the the IMR peak luminosity that the QNM amplitudes for this model have been calibrated to
    # lmn = [ (2,2,0), (2,2,1) ]

    from numpy import linspace
    t = linspace(calibration_T0,20*calibration_T0,1e3)

    # This class is all about the methods
    def __init__(this):
        return None

    #Eval single QNM for mmrdns in the TIME DOMAIN
    @staticmethod
    def meval_mode( l,m,n,                # QNM indeces
                    eta,                  # Symmetric mass ratio
                    params = None,        # Dictionary of optional parameters
                    verbose = False,      # let the people know
                    kind=None,            # strain or psi4
                    plot = False ):

        # Import usefult things
        from numpy import linspace,array,exp,log
        from kerr.formula.ksm2_cw import CW as cwfit
        from kerr.formula.mmrdns_Mfjf import jf as jffit
        from kerr.formula.mmrdns_Mfjf import Mf as Mffit
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            from kerr import rgb,pylim,lim
            import matplotlib as mpl
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16

        # handle empty parameter input
        if params is None: params = {}
        # get the amplitude
        A = mmrdns.Afit(l,m,n,eta)
        # Get the final spin and mass
        jf,Mf = jffit(eta),Mffit(eta)
        # Get the mode frequencies, allow for input of deviations
        cw = cwfit[(l,m,n)](jf) + ( 0 if not ('dcw' in params.keys()) else params['dcw'] )
        # Scale the ferquency by the final mass to get appropriate Geometric values
        cw /= Mf

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Handle input type
        if kind is None:
            kind = 'strain'
        kind = kind.lower()
        if kind == 'psi4':
            # multiple by (1j*cw)^2 to effect two time derivatives
            A = - A * (cw*cw)
        elif kind == 'strain':
            # Do nothing
            _ = None
        else:
            msg = 'kind must be "psi4" or "strain" (case insensitive)'
            error(msg,'mmrdns.meval_mode')
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Construct the waveform as a lambda function.#
        # NOTE that the Amplitude are relative to 10M after the peak liminosity
        t_reff = 0 if not ('t_reff' in params.keys()) else params['t_reff']
        h = lambda t: A * exp( 1j * cw * (t+t_reff-mmrdns.calibration_T0) )
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh_{%i%i%i}(t)/M$'%(l,m,n),fontsize=fs)
            title( r'MMRDNS, $(q,\eta,j_f,M_f) = (%1.4f,%1.4f,%1.4f,%1.4f,)$'%(mmrdns.eta2q(eta),eta,jf,Mf) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')

        # Return the ringown waveform FUNCTION for this QNM
        return h


    @staticmethod
    def meval( theta,           # Source inclination
               phi,             # Orbital phase of system at observation
               eta,             # Symmetric mass ratio
               params = None,   # Bag of optional parameters
               verbose=True,    # Toggle to let the people know
               gwfout=False,    # Toggle for output of nrutils gwf object
               spherical=False, # Toggle to use only spherical modes
               kind=None,       # strain or psi4
               plot = False  ): # Plotting

        # Import needed things
        from numpy import array,linspace,zeros,sum
        from kerr.formula.mmrdns_Mfjf import jf as jffit
        from kerr.pttools import slm
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            import matplotlib as mpl
            from kerr import rgb,pylim,lim
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16

        # Define list of modes to use
        lmn = mmrdns.lmn

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Define the model as a lambda function     #
        jf = jffit(eta)                             #
        # h = lambda t: sum( array( [ slm(jf,l,m,n,theta,phi,norm=True)*mmrdns.meval_mode(l,m,n,eta,params,verbose,kind=kind)(t) for l,m,n in lmn ] ), axis=0 )
        def h(t):

            # Allow for full spheroidal OR spherical model output
            if not spherical:
                h_ = sum( array( [ slm(jf,l,m,n,theta,phi,norm=True)*mmrdns.meval_mode(l,m,n,eta,params,verbose,kind=kind)(t) for l,m,n in lmn ] ), axis=0 )
            else:
                from kerr import sYlm
                llmm = [ (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5) ]
                h_ = sum( array( [ sYlm(-2,ll,mm,theta,phi)*mmrdns.meval_spherical_mode(ll,mm,eta,params,verbose=verbose,kind=kind)(t) for ll,mm in llmm ] ), axis=0 )
            if gwfout:
                from nrutils import gwf
                wfarr = array( [t,h_.real,h_.imag] ).T
                h_ = gwf( wfarr, kind=r'$rh(t,\theta,\phi)/M$' )
            return h_
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh(t,\theta,\phi)/M$',fontsize=fs)
            title( r'MMRDNS, $(q,\eta,j_f,M_f,\theta,\phi) = (%1.4f,%1.4f,%1.4f,%1.4f)$'%(mmrdns.eta2q(eta),eta,theta,phi) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')


        #
        return h

    # Get DIMENSIONFUL complex frequency given a mass ratio and mode index
    @staticmethod
    def cwfit(l,m,n,eta,params=None):

        #
        from kerr.formula.ksm2_cw import CW as cwf
        from kerr.formula.mmrdns_Mfjf import jf as jffit
        from kerr.formula.mmrdns_Mfjf import Mf as Mffit

        # Handle params input
        if params is None: params = {}
        # Get the final spin and mass
        jf,Mf = jffit(eta),Mffit(eta)
        # Get the mode frequencies, allow for input of deviations
        cw = cwf[(l,m,n)](jf) + ( 0 if not ('dcw' in params.keys()) else params['dcw'] )
        # Scale the ferquency by the final mass to get appropriate Geometric values
        cw /= Mf

        #
        return cw


    #Define default fitting functions for the QNM Amplitudes
    # Write function in MATLAB
    @staticmethod
    def Afit(l,m,n,eta):
        # Load the fit functions that have been output by MATLAB -- NOTE that there should is a review ipython notebook that checks these functions by comaring to fit data
        from kerr.formula.mmrdns_amplitudes import A_strain as A
        from numpy import exp,pi
        # It's useful to only calculate the abs of m once
        _m_ = abs(m)
        # If the requested QNM is modeled, then output the related strain amplitude.
        if (l,_m_,n) in A:
            if m>0: # The fit functions are for m>0 ...
                Ak = A[(l,_m_,n)](eta)
            else:   # ... But m<0 is handled by symmetry.
                Ak = A[(l,_m_,n)](eta).conj()
        else:
            # Otherwise, raise an error.
            from kerr import warning
            msg = 'The requested QNM, %s, is not available within MMRDNS. Zero will be returned.'%[l,m,n]
            warning(msg,'mmrdns.Afit')
            Ak = 0.0

        # NOTE that the MATLAB code used to perform the fitting uses a different convention when handling the real and imaginary parts of psi4 than we will use here. The conjugation below makes the output of MMRDNS consistent with nrutils, which injects no manual minus signs when handling psi4, but enforces a phase convention: m>0 has frequencies >0 (non-precessing). NOTE that this may change in the future if significantly precessing systems are found to not sufficiently obey this property. See https://github.com/llondon6/nrutils_dev/blob/master/nrutils/core/nrsc.py#L1714-L1728 for more details.
        Ak = Ak.conj()

        #
        return Ak

    #Function to convert masses to symmetric mass ratio
    @staticmethod
    def m1m2q(m1,m2):
        return mmrdns.q2eta( float(m1)/m2 )

    #Function to convert mass ratio to symmetric mass ratio
    @staticmethod
    def q2eta(q):
        q = float(q)
        return q / (1.0+2*q+q*q)

    # Funcion to convert eta to mass ratio
    @staticmethod
    def eta2q(eta):
        from numpy import sqrt
        b = 2.0 - 1.0/eta
        q  = (-b + sqrt( b*b - 4.0 ))/2.0
        return q

    #
    @staticmethod
    def meval_spherical_mode( ll,mm,               # NOTE that inputs are SPHERICAL harmonic indeces
                              eta,
                              params=None,
                              plot=False,
                              gwfout=False,
                              kind=None,            #
                              mode=None,
                              verbose=False):
        #
        from kerr.formula.mmrdns_Mfjf import jf as jffit
        from kerr import ysprod
        from numpy import zeros,array
        # Setup optional plotting
        if plot:
            from matplotlib.pyplot import plot,xlabel,ylabel,title,gca,figure,xlim
            from kerr import rgb,pylim,lim
            from kerr.formula.mmrdns_Mfjf import Mf as Mffit
            import matplotlib as mpl
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 16
        # handle empty parameter input
        if params is None: params = {}
        #
        jf = jffit(eta)
        #
        lmax = max( [k[0] for k in mmrdns.lmn] )
        l_range = range(2,lmax+1)
        #
        if mode is not None:
            mode_space = [mode]
        else:
            mode_space = mmrdns.lmn
        #
        def h(t):
            #
            ans = zeros( t.shape, dtype=complex )
            #
            for k in mode_space:
                #
                l,m,n = k
                cjk = ysprod( jf, ll,mm, k )
                yk_t = mmrdns.meval_mode(l,m,n,eta,params=params,verbose=verbose,kind=kind)(t)
                #
                ans += cjk * yk_t
            #
            return ans

        #
        if plot:
            figure( figsize=2*array((5,3)) )
            t = mmrdns.t
            y = h(t)
            clr = rgb(3,jet=True)
            plot( t, y.real )
            plot( t, y.imag )
            plot( t, abs(y), color = 'k'  )
            plot( t,-abs(y), color = 'k'  )
            fs = 20
            xlabel(r'$(t-t_{L_{max}})/M$',fontsize=fs)
            ylabel(r'$rh_{%i%i}(t)/M$ Spherical'%(ll,mm),fontsize=fs)
            title( r'MMRDNS, $(q,\eta,j_f,M_f) = (%1.4f,%1.4f,%1.4f,%1.4f,)$'%(mmrdns.eta2q(eta),eta,jf,Mffit(eta)) )
            xlim(lim(t))
            gca().set_yscale("log", nonposy='clip')


        #
        ans = h
        if gwfout:
            from nrutils import gwf
            def gwfh(t):
                h_ = h(t)
                wfarr = array([ t, h_.real, h_.imag ]).T
                return gwf( wfarr, kind=r'$rh_{%i%i}(t)/M$'%(ll,mm) )
            ans = gwfh

        #
        return ans

    # Convert old-style qnm ID into vector: l m n p, where p is the sign of jf relative to initial L and determines the perturbation grade (i.e. prograde or retrograde)
    @staticmethod
    def calc_z(id):

        #
        from numpy import ceil,zeros,floor,log10
        from kerr import error

        #
        a = id

        # The expected length of the encoded vector
        m = ceil(log10(a))/2

        # Name a holder for the decoded vector
        b = zeros( (int(m),) )

        # Name a holder for the signs
        c = zeros(b.size)

        # Decode the signs
        y = a
        for i in range(len(b)):
            #
            c[i] = floor( y/(10**(2*m-i-1)) )
            y = y - c[i]*(10**(2*m-i-1))

        c = (-1)**c

        # Decode the elements' magintudes
        x = y
        for i in range(len(b)):
            #
            b[i] = floor( x/(10**(m-i-1)) )
            x = x - b[i]*(10**(m-i-1))

        # Calculate the recovered vector:
        z = b*c
        z = [ int(k) for k in z ]

        # Reshape the vector into the standard input format for
        # the qnm_object class
        order = m/4
        if order!=round(order):
            error('Input may not be the output of the function "calc_id" within the qnm_object class! It''s not the right size. O.o')

        #
        # z.reshape(4,order).T

        return z
