''' QNM fitting functions. This file was created by bhspec_python_fit_eqns_fun.m'''
# NOTE that these functions give QNM amplitudes in STRAIN.

# Import useful things
from numpy import sqrt,exp

# Write a dictionary of QNM amplitude fitting functions
A_strain = { 
	(2,2,0): lambda eta:  (  0.95846504*exp(2.99318408*1j)*eta + 0.47588079*exp(0.82658128*1j)*(eta**2) + 1.23853419*exp(2.30528861*1j)*(eta**3)  ), 
	(2,2,1): lambda eta:  (  0.12750415*exp(0.05809736*1j)*eta + 1.18823931*exp(1.51798243*1j)*(eta**2) + 8.27086561*exp(4.42014780*1j)*(eta**3) + 26.23294960*exp(1.16782950*1j)*(eta**4)  ), 
	(3,2,0): lambda eta:  (  0.19573228*exp(0.54325509*1j)*eta + 1.58299638*exp(4.24509590*1j)*(eta**2) + 5.03380859*exp(1.71003281*1j)*(eta**3) + 3.73662711*exp(5.14735754*1j)*(eta**4)  ), 
	(4,4,0): lambda eta:  (  0.25309908*exp(5.16320109*1j)*eta + 2.40404787*exp(2.46899414*1j)*(eta**2) + 14.72733952*exp(5.56235208*1j)*(eta**3) + 67.36237809*exp(2.19824119*1j)*(eta**4) + 126.58579931*exp(5.41735031*1j)*(eta**5)  ), 
	(2,1,0): lambda eta: sqrt(1-4*eta)  * (  0.47952344*exp(5.96556090*1j)*eta + 1.17357614*exp(3.97472217*1j)*(eta**2) + 1.23033028*exp(2.17322465*1j)*(eta**3)  ), 
	(3,3,0): lambda eta: sqrt(1-4*eta)  * (  0.42472339*exp(4.54734400*1j)*eta + 1.47423728*exp(2.70187807*1j)*(eta**2) + 4.31385024*exp(5.12815819*1j)*(eta**3) + 15.72642073*exp(2.25473854*1j)*(eta**4)  ), 
	(3,3,1): lambda eta: sqrt(1-4*eta)  * (  0.14797161*exp(2.03957081*1j)*eta + 1.48738894*exp(5.89538621*1j)*(eta**2) + 10.16366839*exp(3.28354928*1j)*(eta**3) + 29.47859659*exp(0.81061521*1j)*(eta**4)  ), 
	(4,3,0): lambda eta: sqrt(1-4*eta)  * (  0.09383417*exp(2.30765661*1j)*eta + 0.82734483*exp(6.10053234*1j)*(eta**2) + 3.33846327*exp(3.87329126*1j)*(eta**3) + 4.66386840*exp(1.75165690*1j)*(eta**4)  ), 
	(5,5,0): lambda eta: sqrt(1-4*eta)  * (  0.15477314*exp(1.06752431*1j)*eta + 1.50914172*exp(4.54983062*1j)*(eta**2) + 8.93331690*exp(1.28981042*1j)*(eta**3) + 42.34309620*exp(4.10035598*1j)*(eta**4) + 89.19466498*exp(1.02508947*1j)*(eta**5)  )
}
# Cleaning up
# del sqrt,exp
