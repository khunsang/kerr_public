# Import useful things
from numpy import exp,sqrt

# Define a dictionary to hold the lambda function for each QNM
A_strain_np = {
			  ((2, 1), (2, 1, 0)): lambda eta,chi_s: eta*sqrt(1-4.0*eta)*(23818489.868391051888*exp(1.101724142877j)*(eta*eta*eta) + 256197217.497183740139*exp(4.243289396938j)*(eta*eta*eta*eta) + 669968995.338050246239*exp(1.101680416418j)*(eta*eta*eta*eta*eta) ),
			  ((2, 2), (2, 2, 1)): lambda eta,chi_s: eta*(620.209517930725*exp(1.219215065165j)*(chi_s) + 2679.665065463591*exp(4.343215098864j)*(eta*chi_s) ),
			  ((2, 2), (2, 2, 0)): lambda eta,chi_s: eta*(-55.592466828217 + -53.202733810311*(chi_s) + 1303.251989077987*(eta) + 11705.828331956920*exp(3.141592653590j)*(eta*eta) + 8079.332358419897*exp(6.283185307180j)*(eta*eta*chi_s) + 44784.903359886455*exp(0.000000000000j)*(eta*eta*eta) + 104.053371156209*(chi_s*chi_s*chi_s) + -1110.877424466633*(eta*chi_s*chi_s*chi_s) + 62129.866337752523*exp(3.141592653590j)*(eta*eta*eta*eta) + 53077.116009563761*exp(3.141592653590j)*(eta*eta*eta*chi_s) + 2814.323762591630*exp(6.283185307180j)*(eta*eta*chi_s*chi_s*chi_s) + 96457.796367635063*exp(6.283185307180j)*(eta*eta*eta*eta*chi_s) ),
			  ((3, 2), (3, 2, 0)): lambda eta,chi_s: eta*(5.334403875962*exp(4.987446897058j) + 103.933569476065*exp(1.849141995234j)*(eta) + 612.375602709867*exp(4.984962946747j)*(eta*eta) + 1124.414902969839*exp(1.834326430052j)*(eta*eta*eta) ),
			  ((3, 3), (3, 3, 0)): lambda eta,chi_s: eta*sqrt(1-4.0*eta)*(5128490.669098326936*exp(6.150130864987j)*(eta*eta*eta) + 55171496.299356497824*exp(3.008505432714j)*(eta*eta*eta*eta) + 144289358.373462617397*exp(6.150073646414j)*(eta*eta*eta*eta*eta) ),
			  ((3, 2), (2, 2, 0)): lambda eta,chi_s: eta*(1.057753651177*exp(3.114897533608j)*(eta) + 3.585930755987*exp(1.542751681244j)*(chi_s) + 5.748773330711*exp(4.906588939122j)*(chi_s*chi_s) + 45.760991448411*exp(4.733799791772j)*(eta*chi_s) + 55.630192745400*exp(2.121315448858j)*(eta*chi_s*chi_s) + 167.033789597905*exp(1.567127183767j)*(eta*eta*chi_s) + 5.362693494306*exp(3.548884462892j)*(chi_s*chi_s*chi_s) + 23.677456310439*exp(0.406973651569j)*(eta*chi_s*chi_s*chi_s) + 148.249036906406*exp(5.533902025652j)*(eta*eta*chi_s*chi_s) + 8.134589129844*exp(0.702080119200j)*(chi_s*chi_s*chi_s*chi_s) + 36.542731634821*exp(3.776429412647j)*(eta*chi_s*chi_s*chi_s*chi_s) + 676.764755932533*exp(4.352971364074j)*(eta*eta*eta*eta*chi_s) ),
			  ((3, 3), (3, 3, 1)): lambda eta,chi_s: eta*sqrt(1-4.0*eta)*(264079.931600457057*exp(1.958281869765j) + 9735022.449579520151*exp(5.099470043710j)*(eta) + 120047686.787852719426*exp(1.957621576619j)*(eta*eta) + 603780165.132755994797*exp(5.099082035289j)*(eta*eta*eta) + 1063837511.040132999420*exp(1.957442976750j)*(eta*eta*eta*eta) ),
			  ((4, 3), (3, 3, 0)): lambda eta,chi_s: eta*sqrt(1-4.0*eta)*(780.191809477672*exp(5.907764528622j)*(eta*chi_s) ),
			  ((4, 4), (4, 4, 0)): lambda eta,chi_s: eta*(0.476729204650*exp(4.889612442241j) + 9.438478682990*exp(1.957457708009j)*(eta) + 31.300357539032*exp(5.177781934595j)*(eta*eta) + 0.610147332323*exp(5.423581458034j)*(chi_s*chi_s*chi_s*chi_s) ),
			  ((4, 3), (4, 3, 0)): lambda eta,chi_s: eta*sqrt(1-4.0*eta)*(1243271.746182312025*exp(1.604312515811j)*(eta*eta*eta) + 13372667.084736810997*exp(4.745894487872j)*(eta*eta*eta*eta) + 34969515.150107681751*exp(1.604292193467j)*(eta*eta*eta*eta*eta) )
			  }
