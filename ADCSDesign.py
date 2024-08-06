import numpy as np
import copy
from OrbitCalculations import *
from SCDesignUtils import *

def setADCSType(payloads, orbit):
    """
    This function sets the ADCS type based on the payload resolution and orbit type
    """
    minRes = min([x.resolution for x in payloads])
    if minRes >= 20 and (orbit == 'LEO' or orbit == 'SSO'):
        ADCSType = "gravGradient"
    elif minRes >= 2:
        ADCSType = "spinner"
    elif minRes < 2:
        ADCSType = "threeAxis"
    return ADCSType
    
def estimateDragCoef(dims):
    """
    This function estimates the drag coefficient from the 
    dimensions of the satellite based on the article by Wallace et al
    Refinements in the Determination of Satellite Drag Coefficients: 
    Method for Resolving Density Discrepancies
    """
    # Change to np.max and np.min if dims is a np array
    LovD = max(dims)/min(dims)
    if LovD > 3.0:
        dragCoef = 3.3
    else:
        dragCoef = 2.2
    return dragCoef

def estimateResidualDipole():
    """
    Anywhere between 0.1 and 20Am^2, 1Am^2 for a small satellite
    """
    resDipole = 5.0
    return resDipole

def gravityGradientTorque(Iy,Iz,a,offNadir,mu):
    """
    This function computes the gravity gradient distrubance torque.
    See SMAD Page 367. Verified OK 9/18/12
    """
    torque = 1.5 * mu * 1/(a**3) * (Iz-Iy) * np.sin(offNadir)
    return torque

def aeroTorque(Cd,As,a,cpacg,mu,rad):
    """
    This function computes the aerodynamic disturbance torque.
    See SMAD page 367. Verified OK 9/18/12.
    """
    V = orbitVelocity(a,a,mu)
    rho = atmosphericDensity(rToh(a,rad))
    torque = 0.5*rho*As*V*V*Cd*cpacg
    return torque

def solarPressureTorque(As,q,sunAngle,cpscg):
    """
    This function computes the solar pressure disturbance torque.
    See SMAD page 367. Verified OK 9/18/12.
    """
    torque = (1367/3e8)*As*(1+q)*(np.cos(sunAngle))*cpscg
    return torque

def magneticFieldTorque(D,a):
    """
    his function computes the magnetic field disturbance torque.
    See SMAD page 367. Verified OK 9/18/12.
    """
    torque = 2*7.96e15*D*a**-3
    return torque

def maxDisturbanceTorque(a,offNadir,Iy,Iz,Cd,As,cpacg,cpscg,sunAngle,D,q,mu,rad):
    """
    Finds the maximum torque the spacecraft will experience
    """
    Tg = gravityGradientTorque(Iy,Iz,a,offNadir,mu)
    Ta = aeroTorque(Cd,As,a,cpacg,mu,rad)
    Tsp = solarPressureTorque(As,q,sunAngle,cpscg)
    Tm = magneticFieldTorque(D,a)
    maxTorque = max([Tg,Ta,Tsp,Tm])
    return maxTorque

def estimateAttCtrlMass(h,attCtrlData):
    """
    This function estimates the mass of a RW from its momentum storage capacity.
    It can also be used to estimate the mass of an attitude control system
    """
    mass = 1.5*h**0.6
    
    # for compType in attCtrlData:
        
    return mass

def computeRWMomentum(Td,a,mu):
    """
    This function computes the momentum storage capacity that a RW
    needs to have to compensate for a permanent sinusoidal disturbance torque
    that accumulates over a 1/4 period
    """
    mom = (1/np.sqrt(2))*Td*0.25*orbitPeriod(a,mu)
    return mom

def estimateAttDetMass(accReq):
    """
    This function estimates the mass of the sensor required for attitude determination
    from its knowledge accuracy requirement. It is based on data from BAll Aerospace, 
    Honeywell, and SMAD chapter 10 page 327
    """
    # mass = 10*acc**(-0.316)
    mass = 1*accReq**(-0.316)
    return mass

def chooseAttCtrlComponents(momentum,torque,spacecraft,compInstance,attCtrlData):
    """
    Chooses the attitude control components. Chooses the best of magnetorquer/thruster and 
    reaction wheel/CMG
    """
    compList = []
    totalCtrlMass = 0
    wheelData = {x:attCtrlData[x] for x in ("reaction wheel","CMG")}
    torquerData = {"magnetorquer":attCtrlData["magnetorquer"]}

    wheel = chooseComponent(momentum,"Momentum (Nms)",compInstance,wheelData,"greater") # can be reaction wheel or CMG
    torquer = chooseComponent(torque,"Torque (Nm)",compInstance,torquerData,"greater") # can be magnetorquer or thruster

    if torquer.performance >= torque:
        # 3 Magnetorquers
        compList.append(torquer)
        compList.append(torquer)
        compList.append(torquer)
        totalCtrlMass += 3*torquer.mass

        # Remove ADCS Thruster propellant
        spacecraft.propellantMassADCS = 0
        spacecraft.deltaVADCS = 0
    else:
        # Add ADCS Thruster based on a 4.5N monopropellant thruster
        # Actual thruster size is not as important as propellant mass here
        torquer = compInstance
        torquer.type = "thruster"
        torquer.mass = .49 # kg
        torquer.dimensions = [.418,.025,.025] # m
        torquer.peakPower = 18 # W
        torquer.avgPower = 18 # W
        torquer.performance = 4.5*np.mean(spacecraft.dimensions)/2 # Nm, average dimension of the spacecraft divided by 2 to get moment arm for thruster

        # 4 thrusters

        compList.append(torquer)
        compList.append(torquer)
        compList.append(torquer)
        compList.append(torquer)
        totalCtrlMass += 4*torquer.mass

    # 4 reaction wheels
    compList.append(wheel)
    compList.append(wheel)
    compList.append(wheel)
    compList.append(wheel)
    totalCtrlMass += 4*wheel.mass

    return compList,totalCtrlMass

def chooseAttDetComponents(accReq,compInstance,attDetData):
    """
    Picks the levels of sensors by slowly increasing the requirement
    """
    compList = []
    totalDetMass = 0
    pickData = attDetData
    pickReq = accReq
    while pickReq <= 10: # 10 deg is the least sensitive sensor in the database
        newComp = chooseComponent(pickReq,"Accuracy (deg)",compInstance,pickData,"less")
        # Two of each component
        totalDetMass += 2*newComp.mass
        newCompCopy = copy.deepcopy(newComp)
        compList.append(newCompCopy)
        compList.append(newCompCopy)
        del pickData[newComp.type]
        pickReq *= 33

    return compList,totalDetMass

def momentOfInertia(k,m,r):
    """
    Calculates an arbitrary moment of inertia from mass m, radius r, and constant k
    """
    moi = k*m*r**2
    return moi

def cuboidMOI(mass,dims):
    """
    Calculates the moment of inertia of a cuboid given mass and dimensions
    """
    alongX = momentOfInertia(1/12,mass,dims[0])
    alongY = momentOfInertia(1/12,mass,dims[1])
    alongZ = momentOfInertia(1/12,mass,dims[2])
    Ix = alongY + alongZ
    Iy = alongX + alongZ
    Iz = alongX + alongY
    MOI = [Ix,Iy,Iz]
    return MOI

# def boxMomentOfInertia(m,dims):


def designADCS(payloads,mission,spacecraft,compInstance,ADCSData):
    """
    Returns the mass of the ADCS Systems
    """
    # pull params
    MOI = spacecraft.MOI
    dim = spacecraft.dimensions
    a = mission.a
    offNadir = 2*np.pi/180 # chosen from other examples, examine more closely
    Cd = spacecraft.Cd
    sunAngle = 0.0 # Chosen from other example, examine more closely
    D = spacecraft.resDipole
    minRes = min([x.resolution for x in payloads])
    ADCSAcc = minRes*.15 # Chosen from historical missions (which are wildly inconsistant)
    ADCSKnow = minRes*.003 # Chosen from historical missions (which are wildly inconsistant)
    dryMass = spacecraft.dryMass
    mu = mission.mu
    rad = mission.rad

    # Extract the component catalogs
    attCtrlData = {x: ADCSData[x] for x in ("reaction wheel","CMG","magnetorquer")}
    attDetData = {y: ADCSData[y] for y in ("star tracker","sun sensor","earth horizon sensor","magnetometer")}

    Iy = MOI[1]
    Iz = MOI[2]
    x = dim[0]
    y = dim[1]
    z = dim[2]
    As = x*y
    cpacg = 0.2*x
    cpscg = 0.2*x
    q = 0.6

    torque = maxDisturbanceTorque(a,offNadir,Iy,Iz,Cd,As,cpacg,cpscg,sunAngle,D,q,mu,rad)
    momentum = computeRWMomentum(torque,a,mu)
    attCtrlList,attCtrlMass = chooseAttCtrlComponents(momentum,torque,spacecraft,compInstance,attCtrlData)
    attDetList,attDetMass = chooseAttDetComponents(ADCSKnow,compInstance,attDetData)
    elMass = attCtrlMass + attDetMass
    spacecraft.attCtrls = attCtrlList
    spacecraft.attDets = attDetList
    # strMass = 0.01*dryMass
    # ADCSMass = elMass + strMass
    ADCSMass = elMass
    ADCSComps = [attDetList,attCtrlList]
    return ADCSMass, ADCSComps