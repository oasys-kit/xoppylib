import numpy
import scipy.constants as codata
from xoppylib.crystals.tools import bragg_metrictensor, lorentz
from xoppylib.crystals.tools import bragg_calc2, crystal_fh

toangstroms = codata.h * codata.c / codata.e * 1e10

def mare_calc(descriptor,H,K,L,HMAX,KMAX,LMAX,FHEDGE,DISPLAY,lambda1,deltalambda,PHI,DELTAPHI, material_constants_library, verbose=0):
    """
        Calculates:

      - Spaghetti plots (lambda versis Psi for multiple crystal reflection)

      - The Umweganregung peak location plot (the diffracted wavelength lambda vs. Psi) for a given primary
        reflection,i.e., an horizontal cut of the spaghetti plot.
      - The Glitches spectrum (the negative intensity for versus the wavelength) or a vertical cut of the spaghetti plot.

      Psi is the azimutal angle of totation, i.e., the totation around
        the H vector (main reflection)


     In other words, if a crystal is set with a particular Bragg angle to match a given reflection (inputs: H,K,L) at
     a given wavelength (input: WaveLength), many other (secondary) reflections are excited when the crystal is rotated
     around the azimutal angle Psi, without changing the Bragg angle.

     The plot (WaveLength,Psi) of the possible reflections is calculated and contains all possible reflection curves
     up to a maximum reflection (input: H Max,  K Max, L Max).

     Umweg plot:
     The intersection of these curves with an horizontal line at the wavelength of the primary reflection
     (input: WaveLength) gives the position of the peaks in the unweg plot. The width of each peak depends on the
     pendent of the curve at the intersection. For that, the Psi1 and Psi2 intersection angles with a band of width
     (input: DeltaWaveLength) are calculated. With this width and the intensity of the diffraction line, it is possible
     to compute a Gaussian that "roughly" describe the peak.


     Glitches plot:
     The intersection of these curves with a vertical line at a given Psi gives the position of the peaks in the
     glitches plot. The width of each peak is the difference between the wavelength values for Psi+/-DeltaPsi
     With this width and the intensity of the diffraction line, it is possible to compute a Gaussian that "roughly"
     describe the peak.


    :param descriptor: a valid crystal name for xraylib
    :param H:    the miller index H
    :param K:    the miller index K
    :param L:    the miller index L
    :param HMAX: the maximum miller index H
    :param KMAX: the maximum miller index K
    :param LMAX: the maximum miller index L
    :param FHEDGE: below this edge (structure factor value) the reflections are discarded
    :param DISPLAY:
            0: Create spaghetti plot script
            0: Create spaghetti+Umweg plot scripts
            0: Create spaghetti+Glitches plot scripts
            0: Create spaghetti+Umweg+Glitches plot scripts
    :param lambda1: wavelength in Angstroms for Umweg plot
    :param deltalambda: delta wavelength in Angstroms for Umweg plot
    :param PHI: phi angle in deg for the Glitches plot
    :param DELTAPHI: delta phi angle in deg for the Glitches plot
    :param verbose: set to 1 for a more verbose output
    :return:
    """

    list_of_scripts = []


    cryst = material_constants_library.Crystal_GetCrystal(descriptor)
    # volume = cryst['volume']
    #
    # #test crystal data - not needed
    #
    # print ("  Unit cell dimensions are %f %f %f" % (cryst['a'],cryst['b'],cryst['c']))
    # print ("  Unit cell angles are %f %f %f" % (cryst['alpha'],cryst['beta'],cryst['gamma']))
    # print ("  Unit cell volume is %f A^3" % volume )
    # print ("  Atoms at:")
    # print ("     Z  fraction    X        Y        Z")
    # for i in range(cryst['n_atom']):
    #     atom =  cryst['atom'][i]
    #     print ("    %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']) )
    # print ("  ")


    fhEdge = FHEDGE
    fhMax = -1e0
    fhMaxIndex = -1

    flg_s = 0
    flg_u = 0
    flg_g = 0


    if DISPLAY == 0:
        flg_s = 1
    elif DISPLAY == 1:
        flg_s = 1
        flg_u = 1
    elif DISPLAY == 2:
        flg_s = 1
        flg_g = 1
    elif DISPLAY == 3:
        flg_s = 1
        flg_u = 1
        flg_g = 1


    # ;
    # ; compute the metric tensor in the reciprocal space
    # ;
    ginv = bragg_metrictensor(cryst['a'],cryst['b'],cryst['c'],cryst['alpha'],cryst['beta'],cryst['gamma'])

    # ;
    # ; wavelength (for intersections: unweg pattern)
    # ;
    # lambda1 = LAMBDA # ; for intersections
    # deltalambda = DELTALAMBDA
    lambdas = numpy.array([lambda1-deltalambda, lambda1, lambda1+deltalambda])

    # ;
    # ; phi (for intersections: glitches pattern)
    # ;
    phi = PHI
    deltaPhi = DELTAPHI
    phis = numpy.array([phi-deltaPhi, phi, phi+deltaPhi])

    # ;
    # ; Main reflection
    # ;
    P = numpy.array([H,K,L],dtype=int)
    p2 = (P[0]**2 + P[1]**2 + P[2]**2)
    pn = numpy.sqrt(p2)

    # ;
    # ; Calculate Reference axis (corresponding to phi =0)
    # ; This is a vector perpendicular to P
    # ;
    mm1 = numpy.dot(ginv,P.T)
    mm2 = [mm1[1],-mm1[0],0]
    mm3 = numpy.min(numpy.abs( mm1[numpy.where(mm1 != 0)]  ))
    M0 = (mm2/mm3)

    # ;
    # ; operational reflections (for permutations)
    # ;
    pmax = numpy.array([HMAX,KMAX,LMAX],dtype=float)
    hh = numpy.arange(pmax[0])+1
    hh = numpy.concatenate((-hh,[0],hh))
    kk = numpy.arange(pmax[1])+1
    kk = numpy.concatenate((-kk,[0],kk))
    ll = numpy.arange(pmax[2])+1
    ll = numpy.concatenate((-ll,[0],ll))

    # ;
    # ; calculate the structure needed for intensity calculations
    # ;

    energy = toangstroms/lambda1

    # ;
    # ; first call to bragg_inp, then calculates the intensity of the main reflection
    # ;
    fhInp = bragg_calc2(descriptor,int(P[0]),int(P[1]),int(P[2]),emin=energy-100,emax=energy+100,estep=10.0, material_constants_library=material_constants_library)
    outInt = crystal_fh(fhInp,energy)

    bragg_angle = 180.0 / numpy.pi * numpy.arcsin(lambda1 * 1e-8/2 / fhInp['dspacing'])
    fhMain = outInt["STRUCT"].real
    intMain = lorentz(bragg_angle)*(fhMain**2)

    if verbose:
        print('Main reflection d-spacing [A]: ',fhInp["dspacing"]*1e8)
        print('Main reflection 1/2d=sin(theta)/lambda: ',1.0/(2*fhInp["dspacing"]*1e8))
        print('Main reflection Bragg angle (using lambda Umweg) [DEG]: ',outInt["THETA"]*180/numpy.pi)
        print('Main reflection Lorentz: ',lorentz(outInt["THETA"]*180/numpy.pi))
        print('Main reflection fh (real part): ',fhMain)
        print('Main reflection intensity: ',intMain)
    #
    # ;
    # ; creates abscissas for spaghettis
    # ;
    alpha = numpy.linspace(-90.0,90.0,500)

    # ;
    # ; main loop over permutations on operatinal reflections
    # ;
    out = numpy.zeros((18,15000))
    ngood = 0
    print("MARE: loop over %d reflections..."%(hh.size*kk.size*ll.size))

    norm = lambda vector: numpy.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

    ijk = 0
    for ih in range(hh.size):
        for ik in range(kk.size):
            for il in range(ll.size):
                ijk += 1
                if verbose: print("\n-------------%d-------------,hkl: %d %d %d"%(ijk,hh[ih],kk[ik],ll[il]))
                r = numpy.array((hh[ih],kk[ik],ll[il]),dtype=int)

                rp = (r*P).sum() / p2 * P
                rp2 = (rp[0]**2 + rp[1]**2 + rp[2]**2)
                rpn = numpy.sqrt(rp2)

                p2new = numpy.dot( P , numpy.dot(ginv,P.T))
                rpnew = numpy.dot( r , numpy.dot(ginv,P.T)) / p2new
                rpnew = rpnew * P

                #   ;
                #   ; Alpha0
                #   ;
                cos_alpha0 = ((r-rp)*M0).sum() / norm(r-rp)/norm(M0)
                alpha0rad = numpy.arccos(cos_alpha0)

                #   ; NOTA BENE: alpha0 is calculating using the orthonormal scalar
                #   ; product. Should this be changed using the metric tensor for a
                #   ; generic structure?
                alpha0 = alpha0rad * 180 / numpy.pi

                #   ;
                #   ; k
                #   ;

                knew1 = 0.5 *  ( numpy.dot( r , numpy.dot( ginv , r.T))  - numpy.dot( r , numpy.dot( ginv , P.T)) )
                knew22 = numpy.dot(r , numpy.dot(ginv , r.T)) - numpy.dot(rpnew, numpy.dot(ginv , rpnew.T))
                knew2 = numpy.sqrt( knew22 )
                knew = knew1 / knew2

                if numpy.abs(knew22) > 1e-8:
                    goodRef = 1
                else:
                    goodRef = 0


                #   ;
                #   ; computes intensity
                #   ;

                fhInp = bragg_calc2(descriptor,int(r[0]),int(r[1]),int(r[2]),emin=energy-100,emax=energy+100,estep=10.0,fileout=None,
                                    material_constants_library=material_constants_library)

                fhInp["f1"] *= 0.0
                fhInp["f2"] *= 0.0

                outInt = crystal_fh(fhInp,energy,forceratio=1)

                if outInt["STRUCT"].real < fhEdge:
                    goodRef = 0

                if goodRef == 1:
                    ngood += 1
                    braggAngleUmweg = outInt["THETA"] * 180 / numpy.pi
                    beta = alpha - alpha0
                    y3 = 1.0 / numpy.sqrt( (knew / numpy.cos(beta * numpy.pi / 180))**2 + p2new / 4 )
                    if verbose: print("Bragg angle (for Umweg): %g"%braggAngleUmweg)

                    theta1 = knew**2 / ((1/lambdas)**2 - p2new / 4)
                    if numpy.abs(theta1[1] > 1):
                        theta2 = [-1000,-1000,-1000]
                        theta3 = [-1000,-1000,-1000]
                    else:
                        theta1 = numpy.arccos(numpy.sqrt(theta1))
                        theta1 = theta1*180/numpy.pi
                        theta2 = alpha0 - theta1
                        theta3 = alpha0 + theta1 - 180
                    #     ;
                    #     ; lambda values for phi intervals (for glitches)
                    #     ;
                    lambdaIntersec = 1.0 / numpy.sqrt( (knew/numpy.cos((phis-alpha0)*numpy.pi/180))**2+p2new/4 )

                    if verbose: print("lambdaIntersec:  ",repr(lambdaIntersec))
                    if verbose: print(("d-spacing [A]: %g"%fhInp["dspacing"]))

                    braggAngleGlitches = lambdaIntersec[1]/2/fhInp["dspacing"]/1e8

                    if numpy.abs(braggAngleGlitches) <= 1:
                        braggAngleGlitches = numpy.arcsin(braggAngleGlitches)*180/numpy.pi
                    else:
                        braggAngleGlitches = 0

                    if verbose: print("Bragg angle (for Glitches): %g"%braggAngleGlitches)
                    #     ;
                    #     ; print/store results
                    #     ;
                    out[0,ngood-1]=r[0]
                    out[1,ngood-1]=r[1]
                    out[2,ngood-1]=r[2]
                    out[3,ngood-1]=alpha0
                    out[4,ngood-1]=knew
                    out[5,ngood-1]=p2new/4
                    out[6,ngood-1]=theta2[0]
                    out[7,ngood-1]=theta2[1]
                    out[8,ngood-1]=theta2[2]
                    out[9,ngood-1]=theta3[0]
                    out[10,ngood-1]=theta3[1]
                    out[11,ngood-1]=theta3[2]
                    out[12,ngood-1]=lambdaIntersec[0]
                    out[13,ngood-1]=lambdaIntersec[1]
                    out[14,ngood-1]=lambdaIntersec[2]
                    out[15,ngood-1]=braggAngleUmweg
                    out[16,ngood-1]=braggAngleGlitches
                    out[17,ngood-1]=(outInt["STRUCT"]).real

                    if outInt["STRUCT"].real > fhMax:
                        fhMax = outInt["STRUCT"].real
                        fhMaxIndex = ngood - 1

    if ngood == 0:
        print("Warning: No good reflections found.")
        return None


    out = out[:,0:(ngood)].copy()
    #
    # ;
    # ; common header for scripts
    # ;
    #
    txt0 = ""
    txt0 += "#\n"
    txt0 += "# xoppy/python macro created by MARE \n"
    txt0 += "# xoppy/mare multiple diffraction  \n"
    txt0 += "#  \n"
    txt0 += "# inputs:  \n"
    # txt0 += "# crystal index: %d\n"%(CRYSTAL)
    txt0 += "# crystal name: %s  \n"%(descriptor)
    txt0 += "# Main reflection: %d %d %d\n"%(H,K,L)
    txt0 += "# Max reflections: %d %d %d\n"%(HMAX,KMAX,LMAX)
    txt0 += "# Wavelength = %g A \n"%(lambda1)
    txt0 += "# Delta Wavelength = %g A\n"%(deltalambda)
    txt0 += "# Phi = %g deg \n"%(PHI)
    txt0 += "# Delta Phi = %g deg\n"%(DELTAPHI)
    txt0 += "# Display: %d \n"%(DISPLAY)
    txt0 += "# Using reflections with fh > %d \n"%(fhEdge)
    txt0 += "# \n"
    txt0 += "# Computed parameters: \n"
    txt0 += "# Number of good reflections: %d \n"%(ngood)
    txt0 += "# M vector (corresponding to phi=0)  %d %d %d \n"%(M0[0],M0[1],M0[2])
    txt0 += "# Intensity of main reflection: %g \n"%(intMain)
    txt0 += "# Structure Factor fh of main reflection: %g \n"%(fhMain)
    txt0 += "# Reflection with maximum intensity: \n"
    txt0 += "#            number: %d \n"%(fhMaxIndex)
    txt0 += "#            miller indices: %d %d %d \n"%(
        int(out[0,fhMaxIndex]),int(out[1,fhMaxIndex]),int(out[2,fhMaxIndex])   )
    txt0 += "#            fh value: %g \n"%(fhMax)



    # ;
    # ; plot script with spaghettis
    # ;

    txt = txt0
    txt += "import numpy\n"
    txt += "import matplotlib.pylab as plt\n"
    txt += "import matplotlib.pylab as plt\n"
    txt += "fig = plt.figure()\n"
    txt += "ax = fig.add_subplot(111)\n"
    txt += "parms = {'n':500, 'xmin':-90.,'xmax':90.,'A_or_eV':0,'ymin':0.,'ymax':3.5}\n"
    txt += "alpha = numpy.linspace(parms['xmin'],parms['xmax'],parms['n'],)\n"
    txt += "ytitle='Photon energy [eV]' if parms['A_or_eV'] == 1 else 'Wavelength [A]'\n"
    txt += "plt.title('MARE-spaghetti, Main diffraction: %d %d %d %s')\n"%(H,K,L,descriptor)
    txt += "plt.xlabel('Azimuthal angle [deg]')\n"
    txt += "plt.ylabel(ytitle)\n"

    txt += "lambdas = numpy."+repr(lambdas)+"\n"
    txt += "phis = numpy."+repr(phis)+"\n"
    txt += "yy =12398.419/lambdas if parms['A_or_eV'] == 1 else lambdas\n"

    for i in range(ngood):
        txt += "# --------------------------------\n"
        txt += "# Reflection nr: %d \n"%(i+1)
        txt += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
        txt += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))
        txt += "y3 = 1.0/numpy.sqrt((%g/numpy.cos((alpha-%g)*numpy.pi/180))**2 + %g)\n"%(out[4,i],out[3,i],out[5,i])
        txt += "if parms['A_or_eV'] == 1: y3=12398.419/y3\n"
        txt += "fg = plt.plot(alpha,y3)\n"
        txt += "ilabel = int(numpy.random.rand()*(parms['n']-1))\n"%()
        txt += "ax.text(alpha[ilabel],y3[ilabel],'%d %d %d',color=fg[0].get_color())\n"%(int(out[0,i]),int(out[1,i]),int(out[2,i]))

    txt += "plt.show()\n"

    list_of_scripts.append(txt)
    if verbose: print(txt)

    # ;
    # ; plot macro with umweg pattern
    # ;
    #
    if flg_u:

        txt1 = txt0
        txt1 += "import numpy\n"
        txt1 += "import matplotlib.pylab as plt\n"
        txt1 += "import matplotlib.pylab as plt\n"
        txt1 += "fig = plt.figure()\n"
        txt1 += "ax = fig.add_subplot(111)\n"
        txt1 += "parms = {'n':500, 'xmin':-90.,'xmax':90.,'A_or_eV':0,'ymin':0.,'ymax':0}\n"
        txt1 += "alpha = numpy.linspace(parms['xmin'],parms['xmax'],parms['n'],)\n"
        txt1 += "umweg = alpha*0\n"
        txt1 += "plt.title('MARE-umweg, Main diffraction: %d %d %d %s at %g A')\n"%(H,K,L,descriptor,lambda1)
        txt1 += "plt.xlabel('Azimuthal angle [deg]')\n"
        txt1 += "plt.ylabel('Approximated intensity')\n"

        for i in range(ngood):
            txt1 += "# --------------------------------\n"
            txt1 += "# Reflection nr: %d \n"%(i+1)
            txt1 += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
            txt1 += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))

            intens = out[17,i]**2 *lorentz(out[15,i])
            txt1 += "theta2 = numpy.array([%g,%g,%g])\n"%(out[6,i],out[7,i],out[8,i])
            txt1 += "theta3 = numpy.array([%g,%g,%g])\n"%(out[9,i],out[10,i],out[11,i])

            if numpy.abs(out[8,i]-out[6,i]) > 1e-6:
                ymax = intens/numpy.abs(out[8,i]-out[6,i])
                txt1 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[15,i]))
                txt1 += "umweg +=  (intens/numpy.abs(theta2[2]-theta2[0]))*numpy.exp(-(alpha-theta2[1])**2/numpy.abs(theta2[2]-theta2[0])**2) \n"
            if numpy.abs(out[11,i]-out[9,i]) > 1e-6:
                ymax = intens/numpy.abs(out[8,i]-out[6,i])
                txt1 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[15,i]))
                txt1 += "umweg +=  (intens/numpy.abs(theta3[2]-theta3[0]))*numpy.exp(-(alpha-theta3[1])**2/numpy.abs(theta3[2]-theta3[0])**2) \n"

        txt1 += "plt.plot(alpha,umweg)\n"

        txt1 += "plt.show()\n"
        #
        list_of_scripts.append(txt1)
        if verbose: print(txt1)




    # ;
    # ; plot macro with glitches pattern
    # ;
    if flg_g:

        txt2 = txt0
        txt2 += "import numpy\n"
        txt2 += "import matplotlib.pylab as plt\n"
        txt2 += "import matplotlib.pylab as plt\n"
        txt2 += "fig = plt.figure()\n"
        txt2 += "ax = fig.add_subplot(111)\n"
        txt2 += "parms = {'n':500, 'xmin':0.5,'xmax':3.5,'A_or_eV':0,'ymin':0.,'ymax':0}\n"
        txt2 += "xmin = parms['xmin']\n"
        txt2 += "xmax = parms['xmax']\n"
        txt2 += "if parms['A_or_eV'] == 1: xmin = 12398.419/xmin\n"
        txt2 += "if parms['A_or_eV'] == 1: xmax = 12398.419/xmax\n"

        txt2 += "xx = numpy.linspace(xmin,xmax,parms['n'],)\n"
        txt2 += "yy = xx*0\n"
        txt2 += "plt.title('MARE-glitches, Main diffraction: %d %d %d %s at %g deg')\n"%(H,K,L,descriptor,phis[1])
        txt2 += "xtitle='Wavelength [A]' if parms['A_or_eV']==0 else 'Photon energy [eV]'\n"
        txt2 += "plt.xlabel(xtitle)\n"
        txt2 += "plt.ylabel('Approximated intensity')\n"

        for i in range(ngood):
            txt2 += "# --------------------------------\n"
            txt2 += "# Reflection nr: %d \n"%(i+1)
            txt2 += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
            txt2 += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))
            txt2 += "lambdas = numpy.array([%g,%g,%g])\n"%(out[12,i],out[13,i],out[14,i])
            txt2 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[16,i]))
            if numpy.abs(out[14,i]-out[12,i]) > 1e-6:
                txt2 += "yy = yy + (intens/numpy.abs(lambdas[2]-lambdas[0]))*numpy.exp(-(xx-lambdas[1])**2/numpy.abs(lambdas[2]-lambdas[0])**2)\n"

        txt2 += "plt.plot(xx,-yy)\n"

        txt2 += "plt.show()\n"

        list_of_scripts.append(txt2)
        if verbose: print(txt2)


    return(list_of_scripts)


if __name__ == "__main__":


    #
    # MARE
    #
    # # descriptor,H,K,L,HMAX,KMAX,LMAX,FHEDGE,DISPLAY,lambda1,deltalambda,PHI,DELTAPHI,verbose=1)
    list_of_scripts = mare_calc("Si2",2,2,2,3,3,3,2e-8,3,1.54,0.01,-20.0,0.1)
    for script in list_of_scripts:
        exec(script)


