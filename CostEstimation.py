import numpy as np

class Mission:
    def __init__(self, satelliteDryMass, structureMass, propulsionMass, ADCSMass, avionicsMass, 
                 thermalMass, EPSMass, satelliteBOLPower, satDataRatePerOrbit, lifetime, numPlanes, 
                 numSats, instruments, launchVehicle, **kwargs):
        self.satelliteDryMass = satelliteDryMass
        self.structureMass = structureMass
        self.propulsionMass = propulsionMass
        self.ADCSMass = ADCSMass
        self.avionicsMass = avionicsMass
        self.thermalMass = thermalMass
        self.EPSMass = EPSMass
        self.satelliteBOLPower = satelliteBOLPower
        self.satDataRatePerOrbit = satDataRatePerOrbit
        self.lifetime = lifetime
        self.numPlanes = numPlanes
        self.numSats = numSats
        self.instruments = instruments
        self.launchVehicle = launchVehicle

        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])

class Instrument:
    def __init__(self, trl, mass, avgPower, dataRate, **kwargs):
        self.trl = trl
        self.mass = mass
        self.avgPower = avgPower
        self.dataRate = dataRate

        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])

class LaunchVehicle:
    def __init__(self, height, diameter, cost, **kwargs):
        self.height = height
        self.diameter = diameter
        self.cost = cost

        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])

def apply_NICM(m, p, rb):
    cost = 25600 * ((p / 61.5) ** 0.32) * ((m / 53.8) ** 0.26) * ((1000 * rb / 40.4) ** 0.11)
    cost = cost / 1.097  # correct for inflation and transform to $M
    return cost

def estimate_instrument_cost(instrument):
    m = instrument.mass
    p = instrument.avgPower
    rb = instrument.dataRate
    
    cost = apply_NICM(m, p, rb)
    
    instrument.cost = cost

def estimate_payload_cost(mission):
    instruments = mission.instruments
    
    costs = 0
    for instrument in instruments:
        estimate_instrument_cost(instrument)
        costs += instrument.cost
    
    total_cost = costs
    total_cost = total_cost / 1000  # to $M
    
    mission.payloadCost = total_cost
    mission.payloadNonRecurringCost = total_cost * 0.8
    mission.payloadRecurringCost = total_cost * 0.2

def estimate_bus_non_recurring_cost(mission):
    strm = mission.structureMass
    prm = mission.propulsionMass
    adcm = mission.ADCSMass
    comm = mission.avionicsMass
    thm = mission.thermalMass
    p = mission.satelliteBOLPower
    epsm = mission.EPSMass
    
    str_cost = 157 * (strm ** 0.83)
    prop_cost = 17.8 * (prm ** 0.75)
    adcs_cost = 464 * (adcm ** 0.867)
    comm_cost = 545 * (comm ** 0.761)
    therm_cost = 394 * (thm ** 0.635)
    pow_cost = 2.63 * ((epsm * p) ** 0.712)
    
    cost = str_cost + prop_cost + adcs_cost + comm_cost + therm_cost + pow_cost
    mission.busNonRecurringCost = cost

def estimate_bus_TFU_recurring_cost(mission):
    strm = mission.structureMass
    prm = mission.propulsionMass
    adcm = mission.ADCSMass
    comm = mission.avionicsMass
    thm = mission.thermalMass
    epsm = mission.EPSMass
    
    str_cost = 13.1 * strm
    prop_cost = 4.97 * (prm ** 0.823)
    adcs_cost = 293 * (adcm ** 0.777)
    comm_cost = 635 * (comm ** 0.568)
    therm_cost = 50.6 * (thm ** 0.707)
    pow_cost = 112 * (epsm ** 0.763)
    
    cost = str_cost + prop_cost + adcs_cost + comm_cost + therm_cost + pow_cost
    mission.busRecurringCost = cost

def estimate_spacecraft_cost_dedicated(mission):
    busnr = mission.busNonRecurringCost
    bus = mission.busRecurringCost
    payl = mission.payloadCost
    
    spacecraftnr = busnr + (payl * 0.6)
    spacecraft = bus + (payl * 0.4)
    sat = spacecraftnr + spacecraft
    
    mission.spacecraftNonRecurringCost = spacecraftnr
    mission.spacecraftRecurringCost = spacecraft
    mission.busCost = busnr + bus
    mission.satelliteCost = sat

def estimate_integration_and_testing_cost(mission):
    scnr = mission.spacecraftNonRecurringCost
    m = mission.satelliteDryMass
    
    iatnr = 989 + (scnr * 0.215)
    iatr = 10.4 * m
    iat = iatr + iatnr
    
    mission.IATNonRecurringCost = iatnr
    mission.IATRecurringCost = iatr
    mission.IATCost = iat

def estimate_program_overhead_cost(mission):
    scnr = mission.spacecraftNonRecurringCost
    scr = mission.spacecraftRecurringCost
    
    prognr = 1.963 * (scnr ** 0.841)
    progr = 0.341 * scr
    prog = progr + prognr
    
    mission.programNonRecurringCost = prognr
    mission.programRecurringCost = progr
    mission.programCost = prog

def estimate_operations_cost(mission):
    sat = mission.satelliteCost
    prog = mission.programCost
    iat = mission.IATCost
    rbo = mission.satDataRatePerOrbit
    life = mission.lifetime
    
    total_cost = sat + prog + iat
    total_cost = total_cost * 0.001097  # correct for inflation and transform to $M
    
    ops_cost = 0.035308 * (total_cost ** 0.928) * life  # NASA MOCM in FY04$M
    ops_cost = ops_cost / 0.001097  # back to FY00$k
    
    if rbo > (5 * 60 * 700 / 8192):
        pen = 10.0
    else:
        pen = 1.0
    
    mission.operationsCost = ops_cost * pen

def get_instrument_list_trls(instruments):
    trls = []
    for instrument in instruments:
        trl = instrument.trl
        trls.append(trl)
    return trls

def compute_cost_overrun(trls):
    min_trl = 10
    for trl in trls:
        if trl < min_trl:
            min_trl = trl
    rss = 8.29 * np.exp(-0.56 * min_trl)
    return 0.017 + (0.24 * rss)

def estimate_total_mission_cost_with_overruns(mission):
    sat = mission.satelliteCost
    prog = mission.programCost
    iat = mission.IATCost
    ops = mission.operationsCost
    launch = mission.launchVehicle.cost
    ins = mission.instruments
    
    mission_cost = sat + prog + iat + ops + (1000 * launch)
    mission_cost = mission_cost / 1000  # to $M
    
    over = compute_cost_overrun(get_instrument_list_trls(ins))
    
    mission.missionCost = mission_cost * (1 + over)

def estimate_total_mission_cost_with_overruns_when_partnership(mission):
    sat = mission.satelliteCost
    prog = mission.programCost
    iat = mission.IATCost
    ops = mission.operationsCost
    launch = mission.launchVehicle.cost
    payl = mission.payloadCost
    bus = mission.busCost
    ins = mission.instruments
    prt = mission.partnershipType
    
    costs = [payl, bus, (1000 * launch), prog, iat, ops]
    
    mission_cost = np.dot(costs, prt)
    mission_cost = mission_cost / 1000  # to $M
    
    over = compute_cost_overrun(get_instrument_list_trls(ins))
    
    mission.missionCost = mission_cost * (1 + over)

def estimate_total_mission_cost_non_recurring(mission):
    bus = mission.busNonRecurringCost
    payl = mission.payloadNonRecurringCost
    prog = mission.programNonRecurringCost
    iat = mission.IATNonRecurringCost
    
    mission_cost = (bus + payl + prog + iat) / 1000  # to $M
    
    mission.missionNonRecurringCost = mission_cost

def estimate_total_mission_cost_recurring(mission):
    bus = mission.busRecurringCost
    payl = mission.payloadRecurringCost
    prog = mission.programRecurringCost
    iat = mission.IATRecurringCost
    ops = mission.operationsCost
    launch = mission.launchVehicle.cost
    numPlanes = mission.numPlanes
    numSats = mission.numSats
    
    mission_cost = (bus + payl + prog + iat + ops) / 1000  # to $M
    
    S = 0.95  # 95% learning curve, means doubling N reduces average cost by 5%
    N = numSats # different from vassar bc pau's code gives total number of sats as opposed to number of sats per plane
    B = -1 / (np.log(1 / S) / np.log(2))
    L = N ** B
    
    total_cost = L * mission_cost
    
    mission.missionRecurringCost = total_cost + launch

def estimate_lifecycle_mission_cost(mission):
    rec = mission.missionRecurringCost
    nr = mission.missionNonRecurringCost
    
    mission.lifecycleCost = rec + nr

def costEstimationManager(mission):
    # Final Goal lifecycleCost
    estimate_bus_TFU_recurring_cost(mission)
    estimate_bus_non_recurring_cost(mission)
    estimate_payload_cost(mission)
    estimate_spacecraft_cost_dedicated(mission)
    estimate_program_overhead_cost(mission)
    estimate_integration_and_testing_cost(mission)
    estimate_operations_cost(mission)

    # total recurring and nonrecurring cost for total lifecycle cost
    estimate_total_mission_cost_recurring(mission)
    estimate_total_mission_cost_non_recurring(mission)

    # total lifecycle cost
    estimate_lifecycle_mission_cost(mission)

    return mission




