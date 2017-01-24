# Implement Final Mass and Spin Fits from arXiv:1404.3197
jf = lambda ETA: ETA * ( 3.4339 - 3.7988*ETA + 5.7733*ETA**2 - 6.3780*ETA**3 )
Mf = lambda ETA: 1.0 + ETA * ( -0.046297 + -0.71006*ETA + 1.5028*ETA**2 + -4.0124*ETA**3 + -0.28448*ETA**4 )
