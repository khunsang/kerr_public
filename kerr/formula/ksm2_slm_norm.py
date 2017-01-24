'''
Module written by "mmrdns_write_python_slm_norm_fit_eqns.m". Fitting functions for searations constants of gravitatioal purterbations of Kerr.
'''

# Import useful things
from numpy import log,exp

# Domain map jf --> kappa( x =def [jf,l,m] ). 
# NOTE that the function below MUST be consistent with the domain_map of "fit_slm_norms.m"
kappa = lambda x: ( log(2.0-x[0])/log(3.0) )**( 1 / (2.0+x[1]-abs(x[2])) )

# Fit Equations for QNM complex frequencies (CW). Note that while these are the zero-damped modes in the extremal kerr limit, these are the effectively unique solutions throughout non-extremal kerr.
CC = {
    (2,2,0): lambda jf:   7.86366171 - 3.61447483*kappa([jf,2,2]) + 3.48996689*kappa([jf,2,2])**2 - 2.29347705*kappa([jf,2,2])**3 + 0.74425069*kappa([jf,2,2])**4  ,
    (2,-2,0): lambda jf: CC[(2,2,0)](jf).conj(),
    (2,2,1): lambda jf:   7.86298703 - 3.59872285*kappa([jf,2,2]) + 2.88459437*kappa([jf,2,2])**2 - 0.92740734*kappa([jf,2,2])**3 - 0.04445478*kappa([jf,2,2])**4  ,
    (2,-2,1): lambda jf: CC[(2,2,1)](jf).conj(),
    (3,3,0): lambda jf:   3.51631915 + 0.16499714*kappa([jf,3,3]) + 1.30114387*kappa([jf,3,3])**2 - 0.83622153*kappa([jf,3,3])**3 + 0.82020713*kappa([jf,3,3])**4  ,
    (3,-3,0): lambda jf: CC[(3,3,0)](jf).conj(),
    (3,3,1): lambda jf:   3.51530809 + 0.19285707*kappa([jf,3,3]) + 0.96814190*kappa([jf,3,3])**2 - 0.00547882*kappa([jf,3,3])**3 + 0.24982172*kappa([jf,3,3])**4  ,
    (3,-3,1): lambda jf: CC[(3,3,1)](jf).conj(),
    (4,4,0): lambda jf:   1.75389888 + 1.00111258*kappa([jf,4,4]) + 1.55498487*kappa([jf,4,4])**2 - 1.22344804*kappa([jf,4,4])**3 + 1.64621074*kappa([jf,4,4])**4  ,
    (4,-4,0): lambda jf: CC[(4,4,0)](jf).conj(),
    (5,5,0): lambda jf:   0.91349889 + 0.89568178*kappa([jf,5,5]) + 2.54404526*kappa([jf,5,5])**2 - 2.82437113*kappa([jf,5,5])**3 + 3.28143852*kappa([jf,5,5])**4  ,
    (5,-5,0): lambda jf: CC[(5,5,0)](jf).conj(),
    (2,1,0): lambda jf:   3.04393302 - 0.06877527*kappa([jf,2,1]) + 0.87671129*kappa([jf,2,1])**2 - 3.92206769*kappa([jf,2,1])**3 + 8.59631959*kappa([jf,2,1])**4 - 8.52199526*kappa([jf,2,1])**5 + 3.31150324*kappa([jf,2,1])**6  ,
    (2,-1,0): lambda jf: CC[(2,1,0)](jf).conj(),
    (3,2,0): lambda jf:   0.74845717 - 0.08157463*kappa([jf,3,2]) + 1.03748092*kappa([jf,3,2])**2 - 3.27926931*kappa([jf,3,2])**3 + 7.24584503*kappa([jf,3,2])**4 - 7.41316799*kappa([jf,3,2])**5 + 3.06056035*kappa([jf,3,2])**6  ,
    (3,-2,0): lambda jf: CC[(3,2,0)](jf).conj(),
    (4,3,0): lambda jf:   0.39542385 - 0.09918352*kappa([jf,4,3]) + 1.52850262*kappa([jf,4,3])**2 - 5.09932727*kappa([jf,4,3])**3 + 10.95647104*kappa([jf,4,3])**4 - 10.99914124*kappa([jf,4,3])**5 + 4.52212985*kappa([jf,4,3])**6  ,
    (4,-3,0): lambda jf: CC[(4,3,0)](jf).conj()
}

# Cleaning up
# del log,exp
