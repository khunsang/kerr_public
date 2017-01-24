'''
Module written by "mmrdns_write_python_sc_fit_eqns.m". Fitting functions for searations constants of gravitatioal purterbations of Kerr.
'''

# Import useful things
from numpy import log,exp

# Domain map jf --> kappa( x =def [jf,l,m] ).
# NOTE that the function below MUST be consistent with the domain_map of "fit_qnm_Mw.m"
kappa = lambda x: ( log(2.0-x[0])/log(3.0) )**( 1 / (2.0+x[1]-abs(x[2])) )

# Fit Equations for QNM complex frequencies (CW). Note that while these are the zero-damped modes in the extremal kerr limit, these are the effectively unique solutions throughout non-extremal kerr.
SC = {
    (2,2,0): lambda jf:   0.55262405*exp(0.00000000*1j) + 6.54272463*exp(0.24443847*1j)*kappa([jf,2,2]) + 5.94664565*exp(3.88409012*1j)*(kappa([jf,2,2])**2) + 5.39298183*exp(1.01651284*1j)*(kappa([jf,2,2])**3) + 3.58701474*exp(4.53395559*1j)*(kappa([jf,2,2])**4) + 1.36858235*exp(1.57079633*1j)*(kappa([jf,2,2])**5) + 0.18520700*exp(4.71238898*1j)*(kappa([jf,2,2])**6)  ,
    (2,-2,0): lambda jf: SC[(2,2,0)](jf).conj(),
    (2,2,1): lambda jf:   0.55229247*exp(0.00000000*1j) + 7.94074969*exp(0.64081239*1j)*kappa([jf,2,2]) + 12.55567057*exp(4.41980669*1j)*(kappa([jf,2,2])**2) + 13.68518711*exp(1.48039237*1j)*(kappa([jf,2,2])**3) + 10.43884041*exp(4.72599435*1j)*(kappa([jf,2,2])**4) + 4.20731453*exp(1.57079633*1j)*(kappa([jf,2,2])**5) + 0.76232588*exp(4.71238898*1j)*(kappa([jf,2,2])**6)  ,
    (2,-2,1): lambda jf: SC[(2,2,1)](jf).conj(),
    (3,2,0): lambda jf:   8.18542769*exp(6.27603422*1j) + 1.55192720*exp(1.79088081*1j)*kappa([jf,3,2]) + 8.94654695*exp(5.18681710*1j)*(kappa([jf,3,2])**2) + 28.66050158*exp(1.63658858*1j)*(kappa([jf,3,2])**3) + 60.77789497*exp(4.72114050*1j)*(kappa([jf,3,2])**4) + 72.13239907*exp(1.57079633*1j)*(kappa([jf,3,2])**5) + 45.38115278*exp(4.71238898*1j)*(kappa([jf,3,2])**6) + 11.84706755*exp(1.57079633*1j)*(kappa([jf,3,2])**7)  ,
    (3,-2,0): lambda jf: SC[(3,2,0)](jf).conj(),
    (4,4,0): lambda jf:   13.05294185*exp(0.00000000*1j) + 9.23462388*exp(0.14179514*1j)*kappa([jf,4,4]) + 7.09045393*exp(3.69184561*1j)*(kappa([jf,4,4])**2) + 6.46711175*exp(0.89254551*1j)*(kappa([jf,4,4])**3) + 4.96905278*exp(4.43853588*1j)*(kappa([jf,4,4])**4) + 2.62299932*exp(1.57079633*1j)*(kappa([jf,4,4])**5) + 0.58168681*exp(4.71238898*1j)*(kappa([jf,4,4])**6)  ,
    (4,-4,0): lambda jf: SC[(4,4,0)](jf).conj(),
    (2,1,0): lambda jf:   3.10089518*exp(6.25822093*1j) + 2.69208437*exp(1.95853947*1j)*kappa([jf,2,1]) + 16.58575360*exp(4.98423605*1j)*(kappa([jf,2,1])**2) + 57.84090876*exp(1.63720921*1j)*(kappa([jf,2,1])**3) + 118.21761290*exp(4.72674943*1j)*(kappa([jf,2,1])**4) + 135.93985738*exp(1.57079633*1j)*(kappa([jf,2,1])**5) + 82.81742189*exp(4.71238898*1j)*(kappa([jf,2,1])**6) + 20.85173245*exp(1.57079633*1j)*(kappa([jf,2,1])**7)  ,
    (2,-1,0): lambda jf: SC[(2,1,0)](jf).conj(),
    (3,3,0): lambda jf:   5.70465254*exp(0.00000000*1j) + 7.94433155*exp(0.18039136*1j)*kappa([jf,3,3]) + 6.55099749*exp(3.77926384*1j)*(kappa([jf,3,3])**2) + 6.31422768*exp(0.93863733*1j)*(kappa([jf,3,3])**3) + 4.81214531*exp(4.46906976*1j)*(kappa([jf,3,3])**4) + 2.38927043*exp(1.57079633*1j)*(kappa([jf,3,3])**5) + 0.48077965*exp(4.71238898*1j)*(kappa([jf,3,3])**6)  ,
    (3,-3,0): lambda jf: SC[(3,3,0)](jf).conj(),
    (3,3,1): lambda jf:   5.70318420*exp(0.00000000*1j) + 8.94926548*exp(0.49834140*1j)*kappa([jf,3,3]) + 12.70528736*exp(4.31772419*1j)*(kappa([jf,3,3])**2) + 15.63533560*exp(1.39390017*1j)*(kappa([jf,3,3])**3) + 14.19057659*exp(4.66913674*1j)*(kappa([jf,3,3])**4) + 7.33238119*exp(1.57079633*1j)*(kappa([jf,3,3])**5) + 1.53701758*exp(4.71238898*1j)*(kappa([jf,3,3])**6)  ,
    (3,-3,1): lambda jf: SC[(3,3,1)](jf).conj(),
    (4,3,0): lambda jf:   15.28866348*exp(0.00000000*1j) + 0.75297352*exp(0.22048290*1j)*kappa([jf,4,3]) + 3.64936150*exp(0.61644055*1j)*(kappa([jf,4,3])**2) + 8.02530641*exp(4.82756576*1j)*(kappa([jf,4,3])**3) + 12.47205664*exp(1.67334685*1j)*(kappa([jf,4,3])**4) + 10.30282199*exp(4.71238898*1j)*(kappa([jf,4,3])**5) + 3.52885679*exp(1.57079633*1j)*(kappa([jf,4,3])**6)  ,
    (4,-3,0): lambda jf: SC[(4,3,0)](jf).conj(),
    (5,5,0): lambda jf:   22.52292196*exp(0.00000000*1j) + 10.44137664*exp(0.11607502*1j)*kappa([jf,5,5]) + 7.79707643*exp(3.61247422*1j)*(kappa([jf,5,5])**2) + 6.59989026*exp(0.83792606*1j)*(kappa([jf,5,5])**3) + 4.90367451*exp(4.40545635*1j)*(kappa([jf,5,5])**4) + 2.59913853*exp(1.57079633*1j)*(kappa([jf,5,5])**5) + 0.58985077*exp(4.71238898*1j)*(kappa([jf,5,5])**6)  ,
    (5,-5,0): lambda jf: SC[(5,5,0)](jf).conj()
}

# Cleaning up
# del log,exp
