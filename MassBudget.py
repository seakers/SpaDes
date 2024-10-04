import numpy as np
from ADCSDesign import *
from EPSDesign import *
from PropulsionDesign import *
from CommsDesign import *
from LaunchVehicleDesign import *

def estimateBussMassfromPayloadMass(payloadMass):
    """
    This rule computes the dry mass of the bus as a linear function of the payload mass.
     Use a factor 4.0 from TDRS 7, in new SMAD page 952. Note that this rule and 
    import-bus-mass-from-DB are never applicable at the same time.
    """
    busMass = payloadMass*4.0
    return busMass

def estimateSatelliteDryMass(payloadMass,busMass):
    """
    This rule computes the dry mass as the sum of bus and payload mass
    (including antennae mass).
    """
    satDryMass = payloadMass + busMass
    return satDryMass

def estimateSatelliteDimensions(satDryMass):
    """
    Estimate dimensions assuming a perfect cube of size given 
    by average density, see SMAD page 337
    """
    r = 0.25*satDryMass**(1/3)
    return [r,r,r]

def prelimMassBudget(payloadMass):
    """
    Function to calculate the dry satellite mass based on the payload mass
    """
    busMass = estimateBussMassfromPayloadMass(payloadMass)
    prelimMass = estimateSatelliteDryMass(payloadMass, busMass)
    return prelimMass

## The following contains the mass calculation for many subsystems. These calculations are simple multipliers and rules of thumb
## and should be expanded upon and later be moved to their own file

def designAvionics(spacecraft):
    """
    Computes comm subsystem mass using rules of thumb
    """
    # pull params
    dryMass = spacecraft.dryMass

    # avMassCoef = 0.0983
    avMassCoef = 0.04 # Temporary number. Reduced because Comms is now separate but still want to account for onboard computers (might just lump in with comms)
    avMass = dryMass*avMassCoef
    avComps = []
    return avMass, avComps

def designThermal(spacecraft):
    """
    Computes thermal subsystem mass using rules of thumb
    """
    # pull params
    dryMass = spacecraft.dryMass

    thermMassCoef = 0.0607
    thermMass = dryMass*thermMassCoef
    thermComps = []
    return thermMass, thermComps

def designStructure(spacecraft):
    """
    Computes structure subsystem mass using rules of thumb
    """
    # pull params
    dryMass = spacecraft.dryMass

    # structureMassCoef = 0.5462
    structureMassCoef = .242 # number from https://ntrs.nasa.gov/api/citations/20140011472/downloads/20140011472.pdf
    structureMass = dryMass*structureMassCoef
    return structureMass

def designLaunchAdapter(dryMass):
    """
    Computes launch adapter mass as 1% of satellite dry mass
    """
    LAMassCoef = 0.01
    LAMass = dryMass*LAMassCoef
    return LAMass

def massBudget(payloads,mission,spacecraft,compInstance,ADCSData,GSData,LVData):
    """
    Calculates a ground up esimate of the mass budget by summing all the subsystem masses
    """


    propMass, propComps = designPropulsion(spacecraft,compInstance)
    structMass = designStructure(spacecraft)
    EPSMass, EPSComps = designEPS(payloads,mission,spacecraft,compInstance)
    ADCSMass, ADCSComps = designADCS(payloads,mission,spacecraft,compInstance,ADCSData)
    avMass, avComps = designAvionics(spacecraft)
    commsMass, commsComps = designComms(payloads,mission,spacecraft,compInstance) # Add modulation information
    thermMass, thermComps = designThermal(spacecraft)
    LVChoice = designLV(mission,spacecraft,compInstance,LVData)
    payloadMass = sum([payload.mass for payload in payloads])
    newDryMass = propMass + structMass + EPSMass + ADCSMass + avMass + payloadMass + commsMass + thermMass
    subsMass = {"Propulsion Mass":propMass, "Structure Mass":structMass, "EPS Mass":EPSMass, "ADCS Mass":ADCSMass, "Avionics Mass":avMass, 
                "Payload Mass":payloadMass, "Comms Mass":commsMass, "Thermal Mass":thermMass}
    components = {"PayloadComps": payloads, "PropComps": propComps, "EPSComps": EPSComps, "ADCSComps": ADCSComps, 
                  "AvComps": avComps, "CommsComps": commsComps, "ThermComps": thermComps, "LVChoice": LVChoice}
    return newDryMass, subsMass, components