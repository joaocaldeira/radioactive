# -*- coding: utf-8 -*-
##############################################################################
# PHYS 211:  Monte Carlo -- radioactive source monte carlo

# PROGRAM:  
# CREATED:   07/16/2016
# LAST EDIT: 10/03/2016
# AUTHOR:    Joao Caldeira
##############################################################################
import numpy as np
import random
import math
from scipy import loadtxt
import matplotlib.pyplot as plt
from tqdm import tqdm

#@profile
def random_vector(dirz=2*random.random()-1):
    #this function returns a random vector. if dirz is provided, the random
    #vector has an inner product of dirz with the z axis.
    dirth=2*np.pi*random.random()
    return np.array([np.sqrt(1-dirz**2)*np.cos(dirth),np.sqrt(1-dirz**2)*np.sin(dirth),dirz])

#@profile
def source(act=1,Tf=1,axis=[0,0,1],cone=np.pi,posn=np.array([0,0,0]),isotope="Na22"):
    #act should be in microCi, Tf in seconds
    #Tf is the total real time the source is allowed to emit gamma rays
    r=3.7e4*act #conversion from microCi to decays per second
    posn=np.asarray(posn)
    tqdm.write(' ') #prevents first print colliding with progress bars
    
    #program takes a list of produced gammas, along with probabilities
    #on each row, first element is probability, second is energies, third is mode
    #mode: 1 if pair production
    if isotope=="Cs137": possible=[[.851,.6616,0]]
    elif isotope=="Na22": possible=[[1.,.511,1],[1.,1.2745,0]]
    elif isotope=="Ba133": possible=[[.1833,.3029,0],[.07164,.276,0],[.022,.053,0],
                                    [.0894,.3839,0],[.6205,.356,0],[.0262,.0796,0],
                                    [.03406,.081,0]]
    elif isotope=="Bi207": possible=[[.0687,1.7702,0],[.745,1.0637,0],[.9774,.5697,0]]
    elif isotope=="Co57": possible=[[.1068,.13647,0],[.856,.12206,0],[.0916,.014413,0]]
    elif isotope=="In116": possible=[[.0246,1.7358,0],[.1,1.5074,0],[.289,.41686,0],
                                    [.0329,.138326,0],[.562,1.0973,0],[.155,2.1121,0],
                                    [.115,.8187,0],[.844,1.29354,0]]
    else:
        tqdm.write("Isotope not recognised! Using Na22.")
        possible=[[1.,.511,1],[1.,1.2745,0]]

    # Generate a list of all decay events from t=0 to t=t_f
    t = []
    t_next=-math.log(random.random())/r
    while t_next < Tf:
        t.append(t_next)
        t_next-=math.log(random.random())/r
    
    events=[]
    
    #rotate axis so it points along z direction (we will rotate back later)
    #if axis is not along the z direction, we rotate along the cross product
    #between axis and ez.
    axis=axis/np.linalg.norm(axis)
    ez=np.array([0.,0.,1.])
    rot=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rotinv=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    out=np.cross(axis,ez)
    if out.any() ==True:
        sin=np.linalg.norm(out)
        cos=math.sqrt(1-sin**2)
        if axis[2] <0: cos=-cos
        rot,rotinv=rot_mat(out,cos=cos)
    #if out is zero, axis is either along the positive or negative z direction
    #turn it around if it is along the negative z direction.
    elif axis[2]<0:
        rot=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
        rotinv=rot
        
    #if there is a specific cone angle selected, only gamma rays making an angle of
    #less than that angle will be accepted into the events array.
    cos=math.cos(cone)
    
    for i in tqdm(range(len(t)), desc='source progress'):
        for j in range(len(possible)):
            #we draw events from the possible array according to their different
            #probabilities.
            if random.random() > possible[j][0]: continue
        
            dirz=2*random.random()-1
                       
            #note we rotate the direction of the gamma ray back into the lab frame
            #using rotinv.
            if dirz>cos:
                events.append([t[i],possible[j][1],posn,np.dot(rotinv,random_vector(dirz))])
            #if this is a pair production event, we create two gamma rays going
            #in opposite directions.
            if possible[j][2]==1 and -dirz> cos:
                events.append([t[i],possible[j][1],posn,np.dot(rotinv,random_vector(-dirz))])
            
    return events
    
#@profile
def source_compton(act=1,Tf=1,axis=[0,0,1],cone=np.pi/18.,posn=np.array([0,0,0]),isotope="Cs137"):
    #this is a quicker source function when we are interested in a small cone
    #and sources with only one gamma ray energy.
    
    #act should be in microCi, Tf in seconds
    #Tf is the total real time the source is allowed to emit gamma rays
    cos=math.cos(cone)
    r=3.7e4*act*(1-cos)/2. #conversion from microCi to decays per second
    #here we already select only the correct proportion of times to have
    #a gamma ray inside the selected cone.
    #NOTE: this simplification is only valid if the source only has one
    #radioactive mode. For sources that emit more than one gamma ray, this
    #would make timing data wrong. For this reason, we use only Cs137 here.
    
    posn=np.asarray(posn)
    tqdm.write(' ') #prevents first print colliding with progress bars
    
    #program takes a list of produced gammas, along with probabilities
    #on each row, first element is probability, second is energies, third is mode
    #mode: 1 if pair production
    if isotope=="Cs137": possible=[[.851,.6616,0]]
    else:
        tqdm.write("Isotope not recognised! Using Cs137.")
        possible=[[.851,.6616,0]]

    # Generate a list of all decay events from t=0 to t=t_f
    t = []
    t_next=-math.log(random.random())/r
    while t_next < Tf:
        t.append(t_next)
        t_next-=math.log(random.random())/r
    
    events=[]
    
    #rotate axis so it points along z direction (we will rotate back later)
    #if axis is not along the z direction, we rotate along the cross product
    #between axis and ez.
    axis=axis/np.linalg.norm(axis)
    ez=np.array([0.,0.,1.])
    rot=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rotinv=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    out=np.cross(axis,ez)
    if out.any() ==True:
        sin=np.linalg.norm(out)
        cos=math.sqrt(1-sin**2)
        if axis[2] <0: cos=-cos
        rot,rotinv=rot_mat(out,cos=cos)
    #if out is zero, axis is either along the positive or negative z direction
    #turn it around if it is along the negative z direction.
    elif axis[2]<0:
        rot=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
        rotinv=rot
        
    cos=math.cos(cone)
    for i in tqdm(range(len(t)), desc='source progress'):
        #here we call random_vector with an argument such that the vector
        #is inside the selected cone.
        events.append([t[i],possible[0][1],posn,np.dot(rotinv,random_vector(random.random()*(1-cos)+cos))])
            
    return events

#@profile
def cyl_intersect(pos,vel,length,radius):
    #in this function we find the intersection of a line parametrised by
    #pos+vel*t with a cylinder.
    #the centre of the cylinder is taken to be the origin, and the axis of the
    #cylinder should point along the z axis.
    [x0,y0,z0]=pos
    [vx,vy,vz]=vel
    
    #first find z intersections. if there are none, skip.
    if vz==0 and abs(z0) > length: return 0,0
    else:
        tz1=-z0/vz-length/(2*vz)
        tz2=-z0/vz+length/(2*vz)
        if tz1>tz2: tz1,tz2=tz2,tz1
        if tz2 <0: return 0,0
    
    #now work in the x-y plane (where the cylinder cross section is a circle) and
    #find where the path of the particle intersects with the edge of the circle.
    #first check how many intersections there are with the circumference. 
    #we use |v|=1 to solve (x0+vx*t)^2+(y0+vy*t)^2==radius**2
    disc=(radius**2*(1-vz**2)-(vy*x0-vx*y0)**2)
    if disc<0: return 0,0 #no intersection
    elif disc==0:
        #disc=0 means either gamma only touches at a point, in which case either
        #we don't care or the particle is moving parallel to z axis.
        if vx!=0 or vy!=0 or x0**2+y0**2>radius**2: return 0,0
        else:
            t1=tz1
            t2=tz2
    else:
        #now we know there are two intersections. obtain times, then compare with
        #z intersections.
        t1=(-x0*vx-y0*vy-math.sqrt(disc))/(1-vz**2)
        t2=t1+2*math.sqrt(disc)/(1-vz**2)
        if t2 <0 or tz1>t2 or tz2<t1: return 0,0
        if tz1>t1: t1=tz1
        if tz2<t2: t2=tz2
    if t1 <0: t1=0 #particle starts already in cylinder
    
    #the two times of intersection with the edges of the cylinder are returned
    return t1,t2
    
#@profile
def cross_sections(en,xen,twocs,phel,incsc):
    #this function returns the cross sections at the energy of the gamma ray.
    #j is the index of the first energy on the table larger than en.
    j=np.searchsorted(xen,en,side='left')
    if j==0: #if energy is below covered, something is probably wrong, but take first val
        cross=twocs[0]
        #pp=pptt[0]
        pe=phel[0]
        comp=incsc[0]
        tqdm.write('Warning: Energy is lower than the lowest value on the table provided.')
    else: #for values not on table, interpolate linearly
        frac=(en-xen[j-1])/(xen[j]-xen[j-1])
        cross=twocs[j-1]+frac*(twocs[j]-twocs[j-1])
        #pp=(pptt[j-1]+frac*(pptt[j]-pptt[j-1]))/cross (not needed)
        pe=(phel[j-1]+frac*(phel[j]-phel[j-1]))/cross
        comp=(incsc[j-1]+frac*(incsc[j]-incsc[j-1]))/cross
    
    return cross,pe,comp
    
#@profile
def compton(en):
    #this function implements the Klein-Nishina formula.
    enf=0
    while enf==0:
        #first we implement the low energy approximation: up to a constant,
        #the differential cross section in that approximation is given by
        #dP/(d Omega)=(1+cos(theta))
        #since d Omega=dphi dtheta sin(theta),
        #dP/(d theta)=(1+cos(theta))sin(theta)
        #integrating, we get the cdf
        #f(y)=1/8(4-3y-y^3),
        #with y=cos(theta)
        #this can be inverted, which is what we do here.
        dice=random.random() #dice randomises value in cdf
        y=2-4*dice+pow(1+(2-4*dice)**2,1/2.)
        if y>0:y=pow(y,1/3.)
        else: y=-pow(-y,1/3.) #pow function only for positive arguments
        cos=y-1/y #this is cos(theta)
        
        #given a cos, find the ratio between energies:
        enf=1/(1+en/.511*(1-cos))
        
        #rejection sampling: we use that the low energy approximation is always
        #larger than the true KN formula. if result is reject, set enf=0 so
        #cycle continues
        if enf**2*(1/enf+enf-1+cos**2)/(1+cos**2) < random.random(): enf=0
        #otherwise transform enf into actual final energy.
        else: enf*=en
    return enf,cos

#@profile
def rot_mat(axis,angle=0,cos=1):
    # this function implements the matrix for the Rodrigues rotation formula
    # for a rotation matrix about an axis,
    # R=1+sin*K+(1-cos)*K^2, with K the matrix axmat below.
    # it can take either an angle or the cos as argument.
    # if it takes cos, rotations are limited to 180º.
    rotmat=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    if angle != 0:
        cos=math.cos(angle)
        sin=math.sin(angle)
    else:    
        sin=math.sqrt(1-cos**2)
    rotax=axis/math.sqrt(np.dot(axis,axis))
    axmat=np.array([[0,-rotax[2],rotax[1]],[rotax[2],0,-rotax[0]],[-rotax[1],rotax[0],0]])
    rotmat+=sin*axmat
    rotmat+=(1-cos)*np.dot(axmat,axmat)
    rotmatinv=rotmat-2*sin*axmat
    
    return rotmat,rotmatinv
    
#@profile
def rot_vect(vect,axis,angle=0,cos=1):
    # this function implements the matrix for the Rodrigues rotation formula
    # for rotating a vector about an axis k,
    # v_rot=v*cos+(k x v)*sin+k(k . v)(1-cos).
    # it can take either an angle or the cos as argument.
    # if it takes cos, rotations are limited to 180º.
    if angle != 0:
        cos=math.cos(angle)
        sin=math.sin(angle)
    else:    
        sin=math.sqrt(1-cos**2)
    rotax=axis/math.sqrt(np.dot(axis,axis))
    rotated=cos*vect+np.cross(rotax,vect)*sin+(1-cos)*np.dot(rotax,vect)*rotax
    return rotated

#@profile
def scatterer(solid="cylinder",center=[0,0,5],axis=[0,0,1],radius=1,length=1,
                    eventsin=[[0,.511,[0,0,0],[0,0,1]]],material="Al",rho=1):
    scatterergeom=[solid,center,axis,radius,length]
    if solid != "cylinder":
        tqdm.write("This geometry is not supported!")
    sc=np.array(scatterergeom[1]) #note: later we will transform coord so that dc=[0,0,0]; this
                        #is relative to lab frame
    sa=np.array(scatterergeom[2])
    sa=sa/np.linalg.norm(sa)
    sr=scatterergeom[3]
    sl=scatterergeom[4]
    
    #load cross sections data
    #NOTE: assuming file has 10 blank or dataless rows, then the data
    xen,cohsc,incsc,phel,ppnf,ppef,twcs,twocs = loadtxt(material+'.txt', unpack=True,skiprows=10)
    
    if material=="Al": rho=2.70 # Al density, in g/cm^3
    elif material=="Pb": rho=11.34 # Pb density, in g/cm^3
    elif rho==1:
        tqdm.write("The program does not have data for the density of the scatterer.")
        tqdm.write("Using density of 1 g/cm^3.")
        tqdm.write("Please input density as 'rho=?' when calling the function.")
    
    #now rotate coordinates so the axis of the scatterer is along the z axis
    #here we find the necessary transformation
    ez=np.array([0.,0.,1.])
    rot=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rotinv=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    out=np.cross(sa,ez)
    if out.any() ==True:
        sin=np.linalg.norm(out)
        cos=math.sqrt(1-sin**2)
        if sa[2] <0: cos=-cos
        rot,rotinv=rot_mat(out,cos=cos)
    #if out is zero, axis is either along the positive or negative z direction
    #turn it around if it is along the negative z direction.
    elif sa[2]<0:
        rot=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
        rotinv=rot
    
    # and then we apply it to all vectors in the problem (also translate coord so center of scatterer is 0)
    for i in range(len(eventsin)):
        eventsin[i][2]=np.dot(rot,eventsin[i][2]-sc)
        eventsin[i][3]=np.dot(rot,eventsin[i][3])
        
    eventsout=[]
    
    for i in tqdm(range(len(eventsin)), desc='scatterer progress'):
        #for each ray in the list eventsin, we will work out what rays come
        #out of the scatterer
        liverays=[eventsin[i]]
        
        while liverays:
            #first, initialise variables. the last call deletes this ray from list of
            #ones to deal with
            tt=liverays[0][0]
            en=liverays[0][1]
            [x0,y0,z0]=liverays[0][2]
            pos=liverays[0][2]
            [vx,vy,vz]=liverays[0][3]
            vel=liverays.pop(0)[3]
            
            t1,t2=cyl_intersect(pos,vel,sl,sr)
            if t2-t1==0: #if there is no intersection, simply add ray to eventsout
                eventsout.append([tt,en,pos,vel])
                continue
                
            #t1, t2 are now the points of intersection. propagate to t1 automatically
            #d is the distance travelled inside the scatterer
            d=t2-t1
            pos=pos+t1*vel
            
            #now find the total cross section... for now ignore Rayleigh scattering
            cross,pe,comp=cross_sections(en,xen,twocs,phel,incsc)
        
            lamb=cross*rho
            #and find where the particle interacts, if it does.
            dice=random.random()
            if(dice < math.exp(-lamb*d)):
                pos=pos+d*vel #if it doesn't interact, move it to end of scatterer
                eventsout.append([tt,en,pos,vel])
                continue
            else: pos=pos-math.log(dice)/lamb*vel #move position to point of interaction
                
            #now we want to check which type of interaction happened
            dice=random.random()
            if dice<pe:
                continue #if photoelectric, nothing comes out of the scatterer
            elif dice <pe+comp:
                enf,cos=compton(en)
                
                #now we change the direction of the vector, with two rotations
                if vy!=0 or vx!=0: out=np.cross(vel,ez)
                else: out=np.array([0.,1.,0.]) #out is a vector perpendicular to vel
                
                veltmp=rot_vect(vel,out,cos=cos) #this goes to the right cone
                vel=rot_vect(veltmp,vel,angle=2*np.pi*random.random()) #this randomises where in the cone we are
                liverays.append([tt,enf,pos,vel]) 
            else: #pair production
                dirvect=random_vector()
                liverays.append([tt,.511,pos,dirvect])
                liverays.append([tt,.511,pos,-dirvect])
                
    #we should now rotate the events back into the lab frame.
    for i in range(len(eventsout)):
        eventsout[i][2]=np.dot(rotinv,eventsout[i][2])+sc
        eventsout[i][3]=np.dot(rotinv,eventsout[i][3])
    
    return eventsout

#@profile
def detector(solid="cylinder",center=[0,0,5],axis=[0,0,1],radius=1,length=1,
                    eventsin=[[0,.511,[0,0,0],[0,0,1]]],material="NaI",rho=1):
    detectorgeom=[solid,center,axis,radius,length] #for a cylinder: center pos, axis, radius, length
    if solid != "cylinder":
        tqdm.write("This geometry is not supported!")
    dc=np.array(detectorgeom[1]) #note: later we will transform coord so that dc=[0,0,0]; this
                        #is relative to source center
    da=np.array(detectorgeom[2])
    da=da/np.linalg.norm(da)
    dr=detectorgeom[3]
    dl=detectorgeom[4]
    
    #load cross sections data
    #NOTE: assuming file has 10 blank or dataless rows, then the data
    xen,cohsc,incsc,phel,ppnf,ppef,twcs,twocs = loadtxt(material+'.txt', unpack=True,skiprows=10)
    
    if material=="NaI":
        rho=3.67 # NaI density, in g/cm^3
        sigma0=.0318 #FWHM of 7.5% at 662 keV
    elif material=="Ge":
        rho=5.32
        sigma0=5.92e-4 #FWHM of 0.14% at 662 keV
    elif rho==1:
        tqdm.write("The program does not have data for the density or FWHM of the detector material.")
        tqdm.write("Using density of 1 g/cm^3.")
        tqdm.write("Please input density as 'rho=?' when calling the detector function.")
    
    #now rotate coordinates so the axis of the detector is along the z axis
    #here we find the necessary transformation
    ez=np.array([0.,0.,1.])
    rot=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    out=np.cross(da,ez)
    if out.any() ==True:
        sin=np.linalg.norm(out)
        cos=math.sqrt(1-sin**2)
        if da[2] <0: cos=-cos
        rot=rot_mat(out,cos=cos)[0]
    #if out is zero, axis is either along the positive or negative z direction
    #turn it around if it is along the negative z direction.
    #note here we don't need the inverse matrix since no events come out
    elif da[2]<0: rot=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    
    # and then we apply it to all vectors in the problem  (also translate coord so center of detector is 0)
    for i in range(len(eventsin)):
        eventsin[i][2]=np.dot(rot,eventsin[i][2]-dc)
        eventsin[i][3]=np.dot(rot,eventsin[i][3])
        
    detected=[]
    
    #now we are ready to see how much energy each particle leaves in the detector!
    for i in tqdm(range(len(eventsin)), desc='detector progress'):
        #for each ray in the list events, we will work out how much energy
        #is absorbed by the detector
        dumpen=0
        liverays=[eventsin[i]]
        
        while liverays:
            #first, initialise variables. the last call deletes this ray from list of
            #ones to deal with
            tt=liverays[0][0]
            en=liverays[0][1]
            [x0,y0,z0]=liverays[0][2]
            pos=liverays[0][2]
            [vx,vy,vz]=liverays[0][3]
            vel=liverays.pop(0)[3]
            
            t1,t2=cyl_intersect(pos,vel,dl,dr)
            if t2-t1==0: continue #if there is no intersection, nothing is absorbed
            
            #t1, t2 are now the points of intersection. propagate to t1 automatically
            #d is the distance travelled inside the detector
            d=t2-t1
            pos=pos+t1*vel
            
            #now find the total cross section... for now ignore Rayleigh scattering, maybe add it later
            cross,pe,comp=cross_sections(en,xen,twocs,phel,incsc)
        
            lamb=cross*rho
            #and find where the particle interacts, if it does.
            dice=random.random()
            if(dice < math.exp(-lamb*d)): continue
            else: pos=pos-math.log(dice)/lamb*vel #move position to point of interaction
                
            #now we want to check which type of interaction happened
            dice=random.random()
            if dice<pe:
                dumpen+=en #if photoelectric, all energy is absorbed.
            elif dice <pe+comp:
                enf,cos=compton(en)
                dumpen+=en-enf #rest is dumped
                
                #now we change the direction of the vector, with two rotations
                if vy!=0 or vx!=0: out=np.cross(vel,ez)
                else: out=np.array([0.,1.,0.]) #out is a vector perpendicular to vel
                
                veltmp=rot_vect(vel,out,cos=cos) #this goes to the right cone
                vel=rot_vect(veltmp,vel,angle=2*np.pi*random.random()) #this randomises where in the cone we are
                liverays.append([tt,enf,pos,vel]) 
            else:
                dumpen+=en-1.022 #if pair production, emit two rays of 511keV, rest is absorbed
                dirvect=random_vector()
                liverays.append([tt,.511,pos,dirvect])
                liverays.append([tt,.511,pos,-dirvect])
                
                
        if dumpen >0:
            #Gaussian distribution
            #stdev depends on material properties and also changes with E
            #stdev/energy goes as 1/sqrt(E)
            #sigma0 should be stdev at 662 keV
            sigma=sigma0*math.sqrt(.662/dumpen)
            dumpen*=1+sigma*math.sin(2*np.pi*random.random())*math.sqrt(-2*math.log(random.random()))
            if dumpen>0: detected.append(dumpen)
        
    try:
        plt.hist(detected,bins=512)
        plt.show()
    except IndexError:
        tqdm.write('No gamma rays detected! Consider changing the geometry or increasing Tf.')
    
    return detected