import numpy as np

def findPropIsp(delV):
    """
    Gets the Isp based on the delta V using rules of thumb from SMAD
    """
    if delV < 100:
        propType = "coldGas"
        Isp = 60
    elif delV < 1000:
        propType = "monoProp"
        Isp = 220
    elif delV < 2000:
        propType = "biProp"
        Isp = 300
    elif delV >= 2000:
        propType = "ion"
        Isp = 2000
    return Isp

def propMassWet(delV,Isp,mwet):
    """
    Calculates the propellant mass from the wet mass and the delta v
    """
    mprop = mwet*(1 - np.exp(-delV/(Isp*9.81)))
    return mprop

def propMassDry(delV,Isp,mdry):
    """
    Calculates the propellant mass from the dry mass and the delta v
    """
    mprop = mdry*(np.exp(delV/(Isp*9.81)) - 1)
    return mprop

def designPropulsion(spacecraft,compInstance):
    """
    Computes dry AKM mass using rules of thumb: 
    94% of wet AKM mass is propellant, 6% motor
    """
    # pull params
    propellantMass = spacecraft.propellantMassInj

    propMass = propellantMass*(6/94)
    propComps = []
    return propMass, propComps