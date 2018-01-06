##############################################################################
# PHYS 211: Monte Carlo -- file to be run
#
# PROGRAM:  
# CREATED:  08/03/2016
# AUTHOR:   Joao Caldeira
##############################################################################

import radioactive as ra
import numpy as np

events=ra.source_compton(act=10.,Tf=800.,axis=[1.,0,0],posn=[-10.,0,0],cone=np.pi/60.,isotope="Cs137") 
#events=ra.source(act=1,Tf=2.,axis=[1.,0,0],posn=[-5.,0,0],cone=np.pi/3.,isotope="Na22") 
# only rays making an angle of less than cone with axis are accepted. last two
# arguments are optional.

eventsscat=ra.scatterer(solid="cylinder",center=[0,0,0.],axis=[0,0,1.], radius=2.,
                    length=10.,eventsin=events,material="Al")

ra.detector(solid="cylinder",center=[10*np.cos(np.pi/6.),10*np.sin(np.pi/6.),0],
            axis=[np.cos(np.pi/9.),np.sin(np.pi/9.),0],
            radius=2.5,length=5.,eventsin=eventsscat,material="NaI")