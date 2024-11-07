import numpy as np

def dB2lin(x):
    return 10**(x/10)

def lin2dB(x):
    """
    Returns x in Decibels
    """
    return 10*np.log10(x)

def calcRange(h, epsMin):
    """
    
    """
    rad = 6378 # km
    rho = np.arcsin(rad/(rad+h))*180/np.pi
    etaMax = np.arcsin(np.sin(rho*np.pi/180)*np.cos(epsMin*np.pi/180))*180/np.pi
    lambdaMax = 90 - epsMin - etaMax
    dMax = rad*np.sin(lambdaMax*np.pi/180)/np.sin(etaMax*np.pi/180)
    return dMax

def inflate(x1, y1, y2):
    factors = [0.097, 0.088, 0.08, 0.075, 0.078, 0.08, 0.081, 0.084, 0.082, 0.081, 0.081, 0.085, 0.095, 0.1, 0.102, 0.105, 0.113, 0.13, 0.14, 0.138, 0.14, 0.151, 0.154, 0.155, 0.156, 0.156, 0.158, 0.163, 0.168, 0.169, 0.172, 0.174, 0.175, 0.178, 0.18, 0.183, 0.188, 0.194, 0.202, 0.213, 0.225, 0.235, 0.243, 0.258, 0.286, 0.312, 0.33, 0.352, 0.379, 0.422, 0.479, 0.528, 0.56, 0.578, 0.603, 0.625, 0.636, 0.66, 0.687, 0.72, 0.759, 0.791, 0.815, 0.839, 0.861, 0.885, 0.911, 0.932, 0.947, 0.967, 1, 1.028, 1.045, 1.069, 1.097, 1.134, 1.171, 1.171, 1.216, 1.208, 1.226, 1.244, 1.264, 1.285, 1.307, 1.328, 1.35, 1.372, 1.395, 1.418]
    years = []
    year = 1930
    for i in range(len(factors)):
        years.append(year+i)
    
    f1 = factors[years.index(y1)]
    f2 = factors[years.index(y2)]

    return (x1/f1)*f2


def dimPatchAntenna(er, fGHz):
    """
    Returns the dimensions of a patch antenna
    """
    f = fGHz*1e9
    lambdaO = 3e8/f
    h = .1*lambdaO
    W = 3e8/(2*f)*np.sqrt(2/(er+1))
    ereff = (er+1)/2 + (er-1)/(2*np.sqrt(1 + 12*h/W))
    dl = .824*h*((ereff + .3)*(W/h + .264))/((ereff - .258)*(W/h + .8))
    Leff = 3e8/(2*f*np.sqrt(ereff))
    L = Leff - 2*dl
    h = 0.001*max(W,L)
    dims = [W,L,h]
    
    return dims

def patchMass(dims):
    """
    Estimates the mass of the patch antenna using an assumed density and the dimensions
    """
    W = dims[0]
    L = dims[1]
    h = dims[2]
    rho = .0027
    volume = W*L*h
    volumeReal = .5*volume
    return rho*volumeReal

def gain2Diameter(GdB, fGHz, eff):
    """
    Calculates a diameter based on the gain given a certain frequency
    """
    c = 3e8
    if GdB < 0:
        GdB = 0
    Glin = dB2lin(GdB)
    lamb = c/(fGHz*1e9)
    D = np.sqrt((Glin*(lamb**2))/((np.pi**2) * eff))
    h = D/10 ## Chosen Arbitrarily, Find better way to do this
    
    design = [D, D, h]
    return design

def parabolicMass(D):
    """
    Estimates the mass of a parabolic antenna given diameter
    """
    rho = 0
    if D <= .5:
        rho = 20
    elif D <= 1:
        rho = 10
    elif D <= 10:
        rho = 5
    else:
        rho = 5

    h = 1/10
    alpha = .05
    mass = rho*np.pi*((D/2)**2)*(((1+4*(h**2))**1.5) - 1)/(6*(h**2))*(1+alpha)

    return mass

def massCommElectronics(Ptx, dryMass, bandi):
    """
    Estimates the mass of the electronics
    """
    m = 0
    if dryMass < 30:
        if bandi == 0:
            m = .1
        elif bandi == 1:
            m = .2
        elif bandi == 2:
            m = .3
        else:
            m = .3
    else:
        powerTx = Ptx/.1
        powerAmp = Ptx/.7
        massTransmitter = powerTx*.008 + .5
        massAmplifier = powerAmp*.005 + .2

        m = massTransmitter + massAmplifier
    
    return m

def costAntenna(mass, antennaType):
    """
    Estimates the cost of the antenna
    """
    if antennaType == "Dipole":
        cost = 5*mass
    elif antennaType == "Patch":
        cost = 5*mass
    else:
        if .75 <= mass and mass <= 87*1.25:
            nrc = inflate(1015*(mass**.59),1992,2015)
            rc = inflate(20 + 230*(mass**.59),1992,2015)
            cost = nrc + rc
        else:
            cost = 1e10
    
    return cost

def costElectronics(mass, nChannels):
    """
    Estimates the cost of the electronics
    """
    if mass < 14*0.7:
        mass = 14*0.7
    elif mass > 144*1.25:
        mass = 144*1.25
    
    if 14*.07 <= mass and mass <= 144*1.25:
        cost = 917 * mass**.7 + 179*mass
    else:
        cost = 1e10
    
    return cost

def commsPower(Ptx):
    """
    Estimates the peak power used by the comms system
    """
    eff = .45 - .015*(Ptx - 5)

    if eff < .45:
        eff = .45
    
    transceiverPower = Ptx*(1 + 1/eff)
    antennaPower = 0
    othersPower = 0

    peakPowerComm = antennaPower + transceiverPower + othersPower
    avgPowerComm = 0 + 1 + othersPower
    offPowerComm = othersPower

    tPeakComm = .15
    tAvgComm = .6
    tOffComm = .25

    # return tPeakComm*peakPowerComm + tAvgComm*avgPowerComm + tOffComm*offPowerComm
    return peakPowerComm

def designAntenna(alt,dryMass,dataRate,Ptx,Gtx,bandi,compInstance): # maybe rework to reflect whats in SMAD
    """
    Returns an antenna design (taken from dev_Alan branch of VASSAR_lib)
    """
    # Initialized parameters
    bandsNEN = ["UHF", "Sband", "Xband", "Kaband"]
    bwNEN = [[137.825e6,137.175e6],[2.29e9,2.2e9],[8.175e9,8.025e9],[27e9,25.5e9]]
    fNENDL = [1.44e8, 2.30e9, 8.10e9, 2.60e10]
    gNENDL = [18, 50, 56.8, 70.5]
    tgsNENDL = [200, 165, 170, 225]
    fNENUL = [1.39e8, 2.07e9]

    dataPerDay = dataRate*1e6*24*3600
    rbDL = dataPerDay/(2*15*60)
    fDL = 0
    gGS = 0
    tGS = 0

    # Assuming NEN here
    fDL = fNENDL[bandi]
    gGS = gNENDL[bandi]
    tGS = tgsNENDL[bandi]

    lamb = 3e8/fDL
    r = calcRange(alt,15)
    k = 1.38e-28
    fDLGHz = fDL/1e9
    # ebN0min = [10.6 10.6 14 18.3] # Eb/N0 necessary to achieve a good communication for BER=10e-6 and M=2,4,8,16 (M-PSK)
    ebN0min = 10.6
    ebN0 = lin2dB(Ptx*Gtx) + gGS + 2*lin2dB(lamb/(2*np.pi*r)) - lin2dB(k*tGS*rbDL)

    linkBudgetClosed = False
    if(ebN0 > ebN0min):
        Gtx = lin2dB(Gtx)

        # Assuming NEN
        nChannels = (bwNEN[bandi][0] - bwNEN[bandi][0])/32e6
        if Gtx < 3 and bandi == 0:
            lamb = 3e8/fDL
            if Gtx < 1.76: 
                Ltx = .1*lamb
            elif Gtx > 1.76 and Gtx < 2.15:
                Ltx = .5*lamb
            elif Gtx > 2.15 and Gtx < 5.2:
                Ltx = 5/4*lamb
            
            antennaType = "Dipole Antenna"
            antennaDim = [.25,.05,.001]
            massADl = .05
            linkBudgetClosed = True

        elif Gtx < 9 and (bandi == 0 or bandi == 1):
            er = 10.2
            antennaDim = dimPatchAntenna(er, fDLGHz)
            massADl = patchMass(antennaDim)
            W = antennaDim[0]
            L = antennaDim[1]
            antennaType = "Patch Antenna"
            linkBudgetClosed = True
        else:
            antennaDim = gain2Diameter(Gtx, fDLGHz, .6)
            Dtx = antennaDim[0]
            if Dtx > .3 and Dtx < 4.5:
                antennaType = "Parabolic Antenna"
                linkBudgetClosed = True
            elif Dtx <= .3:
                Dtx = .3
                antennaType = "Parabolic Antenna"
                linkBudgetClosed = True
            else:
                return None
            massADl = parabolicMass(Dtx)
            F = .5*Dtx
            H = pow(Dtx, 2) / (16*F)
        
        if linkBudgetClosed:
            massA = massADl
            massE = massCommElectronics(Ptx, dryMass, bandi)*2 + .01*dryMass
            commsMass = massA + massE

            costA = costAntenna(massA, antennaType)
            costE = costElectronics(massE, nChannels)
            costComms = costA + costE

            powerComms = commsPower(Ptx)
        else:
            costComms = 1e10
    else:
        # if Gtx < 3:
        #     lamb = 3e8/fDL
        return None

    antenna = compInstance
    antenna.type = antennaType
    antenna.mass = massA
    antenna.dimensions = antennaDim
    antenna.peakPower = powerComms
    antenna.avgPower = powerComms # Should adjust commsPower to also give avg power
    antenna.cost = costA

    electronics = compInstance
    electronics.mass = massE
    electronics.cost = costE

    return [antenna,electronics] ## Change to return a list of component objects

def designComms(payloads,mission,spacecraft,compInstance):
    """
    Function to design the communication system. Cycles through band and modulation possibilities to find the lowest cost component
    which can fulfill requirements
    """
    # pull params
    dataRate = sum([payload.dataRate for payload in payloads])
    dryMass = spacecraft.dryMass
    alt = mission.h # km

    # Parameters

    nenAntennas = []
    bandsNEN = ["UHF", "Sband", "Xband", "Kaband"]
    recieverPower = [1] * 250 
    antennaGain = [1] * 50
    massMin = 1e10
    bestCommsCompsList = None
    bandMin = -1
    imin = -1
    jmin = -1

    for bandi in range(len(bandsNEN)): ## Eventually change to for band in bands. Its cleaner and more pythony
        bandAntennas = []

        for i in range(len(recieverPower)):
            if i == 0:
                recieverPower[i] = 1
            else:
                recieverPower[i] = recieverPower[i-1] + 1

            powerAntennas = []
            for j in range(len(antennaGain)):
                if j == 0:
                    antennaGain[j] = 1
                else:
                    antennaGain[j] = antennaGain[j-1] + 1

                commsCompList = designAntenna(alt,dryMass,dataRate,recieverPower[i],antennaGain[j],bandi,compInstance)
                powerAntennas.append(commsCompList)
                if not commsCompList == None:
                    [antenna,electronics] = commsCompList
                    newMass = antenna.mass + electronics.mass

                    if newMass < massMin:
                        massMin = newMass
                        bestCommsCompsList = commsCompList
                        # bandMin = bandi
                        # imin = i
                        # jmin = j
                
            # bandAntennas.append(powerAntennas)

        # nenAntennas.append(bandAntennas)
    
    if bestCommsCompsList == None:
        bestCommsCompsList = commsCompList
        massMin = newMass
    # if imin == -1 or jmin == -1:    
    #     bestCommsList = designAntenna(alt,dryMass,dataRate,max(recieverPower),max(antennaGain),bandi,compInstance)
    #     massMin = bestCommsList[0].mass + bestCommsList[1].mass
    # else:
    #     bestCommsList = nenAntennas[bandMin][imin][jmin]
    #     massMin = nenAntennas[bandMin][imin][jmin][0].mass + nenAntennas[bandMin][imin][jmin][1].mass        

    spacecraft.comms = bestCommsCompsList
    return massMin,bestCommsCompsList