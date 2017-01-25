# Kerr

##Low Level Tools for Black Hole Quasi-normal Modes
 * Kerr QNM frequencies
 * Kerr separation constants
 * Spheroidal Harmonics
 * Leaver's continued fraction method
 * Solution space mapping
 * Fit equations for QNM excitations
 * Fit equations for frequencies and separation constants

## How to add to your system's PYTHONPATH
If you have cloned the repository to
```bash
/Users/home/kerr_public/
```

Then you will want to add the following line to your ~/.bash_profile (or .bashrc or equivalent)
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/home/kerr_public/"
```

You will then want to source the related file
```bash
source ~/.bash_profile
```

## Quick Examples
### Get Kerr QNM Frequencies and Damping times using tabulated results of Leaver's Method
```python
from kerr import leaver
l,m,n = 3,3,0
jf = 0.99
# NOTE that jf<0 accesses teh retrograde frequencies 
untiless_qnm_frequency, separation_constant = leaver( jf, l, m, n )
```

 The result is untiless_qnm_frequency = 1.4397416351127201-0.010531060217508836j

### Get Kerr QNM Frequencies and Damping times using fitting formulas
```python
from kerr.formula.ksm2_cw import CW as cwfit
l,m,n = 3,3,0
jf = 0.99
# NOTE that there is a convention difference between the fit and the results of kerr.leaver
untiless_qnm_frequency = cwfit[l, m, n](jf)
```

 The result is untiless_qnm_frequency = 1.4398601232446264+0.010494215308402903j
 
#### A note on the extremal kerr limit $j_f \rightarrow 0$
The above example has been chosen to demonstrate that at near the extremal kerr limit, many QNM frequencies aspmtote to m/2. In the case of (l,m)=(3,3), the extremal frequency is 1.5 (M=c=1). While the tabulated evaluations of leaver's method have not been carried out *exactly* at the extremal limit, the fits enforce this behavior and evaluations of Leaver's method very near the kerr limit have been performed as a check.

## A few working notes
 * [kerr_public/notes/ns/mmrd.pdf](https://github.com/llondon6/kerr_public/blob/master/notes/ns/mmrd.pdf)
 
## Useful ipython notebooks
* Test Spheroidal Harmonic implementation
 [kerr_public/notes/ns/notebooks/test_slm.ipynb](https://github.com/llondon6/kerr_public/blob/master/notes/ns/notebooks/test_slm.ipynb)
* Query how well the QNM fits satisfy Leaver's equations
 [kerr_public/notes/ns/notebooks/test_python_fit_equations.ipynb](https://github.com/llondon6/kerr_public/blob/master/notes/ns/notebooks/test_python_fit_equations.ipynb)
* Explore the QNM solution space using Leaver's continued fraction method
 [kerr_public/examples/leaver_example.ipynb](https://github.com/llondon6/kerr_public/blob/master/examples/leaver_example.ipynb)
