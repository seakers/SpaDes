from SCDesignClasses import Component

def getComponents():

    # Create Components to put in spacecraft. Same as ones used in 
    # Spacecraft Component Adaptive Layout Environment (SCALE): An efficient optimization tool
    # by Fakoor
    # transferLearningComponents = [
    # componentList = [
    # # [
    #     Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    #     Component(type="transmitter", mass=4.5, dimensions=[.25,.15,.05], heatDisp=12),
    #     Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10),
    #     Component(type="reciever", mass=4, dimensions=[.2,.15,.03], heatDisp=11),
    #     Component(type="reciever", mass=3, dimensions=[.175,.125,.03], heatDisp=9),
    #     Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7),
    #     Component(type="OBDH", mass=9, dimensions=[.24,.18,.18], heatDisp=6),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="payload", mass=5, dimensions=[.3,.25,.2], heatDisp=3),
    #     Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5),
    #     Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5)
    # ]
    # componentList = [
    #     # Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2),
    #     # Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10),
    #     Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7)
    # ]
        # ],

    #     # componentTransfer2 = [
    #     [
    #         Component(type="fuel tank", mass=15, dimensions=[.3,.25,.2], heatDisp=3),
    #         Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #         Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #         Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #         Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.2),
    #         Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.2),
    #         Component(type="antenna", mass=3.5, dimensions=[.3,.12,.1], heatDisp=8),
    #         Component(type="antenna", mass=3, dimensions=[.25,.1,.08], heatDisp=7),
    #         Component(type="data recorder", mass=2.5, dimensions=[.22,.18,.06], heatDisp=6),
    #         Component(type="PCU", mass=6.5, dimensions=[.25,.2,.12], heatDisp=6),
    #         Component(type="OBDH", mass=8, dimensions=[.22,.18,.16], heatDisp=5),
    #         Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #         Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #         Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #         Component(type="payload", mass=4.5, dimensions=[.28,.22,.18], heatDisp=2.8),
    #         Component(type="solar panel", mass=1.2, dimensions=[.18,.48,.01], heatDisp=1.4),
    #         Component(type="solar panel", mass=1.2, dimensions=[.18,.48,.01], heatDisp=1.4),
    #         Component(type="battery", mass=6, dimensions=[.2,.18,.12], heatDisp=2)
    #     ],

    #     # componentTransfer3 = [
    #     [
    #         Component(type="OBDH", mass=3, dimensions=[.24,.19,.07], heatDisp=7),
    #         Component(type="PCU", mass=7, dimensions=[.26,.21,.13], heatDisp=6.5),
    #         Component(type="command processor", mass=7.5, dimensions=[.23,.19,.17], heatDisp=5.5),
    #         Component(type="inertial measurement unit", mass=2.3, dimensions=[.14,.11,.09], heatDisp=2.5),
    #         Component(type="inertial measurement unit", mass=2.3, dimensions=[.14,.11,.09], heatDisp=2.5),
    #         Component(type="antenna", mass=4.2, dimensions=[.35,.14,.12], heatDisp=9),
    #         Component(type="antenna", mass=3.2, dimensions=[.22,.09,.07], heatDisp=7.5),
    #         Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #         Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #         Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #         Component(type="payload", mass=5, dimensions=[.3,.24,.2], heatDisp=3),
    #         Component(type="solar panel", mass=1.3, dimensions=[.19,.49,.01], heatDisp=1.5),
    #         Component(type="solar panel", mass=1.3, dimensions=[.19,.49,.01], heatDisp=1.5),
    #         Component(type="battery", mass=5.5, dimensions=[.22,.2,.13], heatDisp=2.2),
    #         Component(type="propellant tank", mass=12, dimensions=[.28,.22,.18], heatDisp=4),
    #         Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9),
    #         Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9),
    #         Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9)
    #     ],

    #     # componentTransfer4 = [
    #     [
    #         Component(type="PCU", mass=6, dimensions=[.25,.2,.12], heatDisp=6.2),
    #         Component(type="OBDH", mass=8.5, dimensions=[.24,.19,.16], heatDisp=5.8),
    #         Component(type="flight computer", mass=7.8, dimensions=[.23,.19,.18], heatDisp=5.6),
    #         Component(type="reaction wheel", mass=2.8, dimensions=[.14,.12,.1], heatDisp=3),
    #         Component(type="reaction wheel", mass=2.8, dimensions=[.14,.12,.1], heatDisp=3),
    #         Component(type="antenna", mass=4.5, dimensions=[.34,.15,.12], heatDisp=9.5),
    #         Component(type="antenna", mass=3.3, dimensions=[.23,.1,.08], heatDisp=7.8),
    #         Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #         Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #         Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #         Component(type="payload", mass=5.2, dimensions=[.29,.24,.21], heatDisp=3.2),
    #         Component(type="solar panel", mass=1.4, dimensions=[.18,.47,.01], heatDisp=1.6),
    #         Component(type="solar panel", mass=1.4, dimensions=[.18,.47,.01], heatDisp=1.6),
    #         Component(type="battery", mass=5.8, dimensions=[.22,.21,.14], heatDisp=2.3),
    #         Component(type="hydrazine tank", mass=13, dimensions=[.29,.23,.19], heatDisp=4.1),
    #         Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1),
    #         Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1),
    #         Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1)
    #     ]
    # ]

    # componentList = [
    # # [
    #     Component(type="reaction wheel", mass=3, dimensions=[.14,.13,.1], heatDisp=3.1),
    #     Component(type="reaction wheel", mass=3, dimensions=[.14,.13,.1], heatDisp=3.1),
    #     Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
    #     Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
    #     Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
    #     Component(type="antenna", mass=4.3, dimensions=[.33,.15,.12], heatDisp=9.7),
    #     Component(type="antenna", mass=3.4, dimensions=[.22,.09,.08], heatDisp=7.9),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
    #     Component(type="solar panel", mass=1.5, dimensions=[.19,.48,.01], heatDisp=1.7),
    #     Component(type="solar panel", mass=1.5, dimensions=[.19,.48,.01], heatDisp=1.7),
    #     Component(type="battery", mass=6, dimensions=[.23,.2,.14], heatDisp=2.5),
    #     Component(type="propellant tank", mass=12.5, dimensions=[.28,.22,.18], heatDisp=4.2),
    #     Component(type="PCU", mass=6.3, dimensions=[.26,.21,.13], heatDisp=6.4),
    #     Component(type="OBDH", mass=8.2, dimensions=[.24,.18,.15], heatDisp=5.9),
    #     Component(type="flight computer", mass=7.6, dimensions=[.22,.19,.17], heatDisp=5.7),
    #     Component(type="payload", mass=5.5, dimensions=[.3,.23,.22], heatDisp=3.4)
    # ]
    # ]

    componentList = [
        Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5),
        Component(type="solar panel", mass=1.6, dimensions=[.21,.52,.01], heatDisp=1.6),
        Component(type="solar panel", mass=1.4, dimensions=[.19,.49,.01], heatDisp=1.4),

        Component(type="payload", mass=6.5, dimensions=[.3,.24,.22], heatDisp=3.2),
        Component(type="payload", mass=5.5, dimensions=[.28,.22,.2], heatDisp=3),

        Component(type="transmitter", mass=3.8, dimensions=[.25,.1,.08], heatDisp=12),
        Component(type="transmitter", mass=4.0, dimensions=[.23,.12,.09], heatDisp=11),

        Component(type="receiver", mass=3.3, dimensions=[.21,.12,.05], heatDisp=9),
        Component(type="receiver", mass=3.5, dimensions=[.22,.13,.06], heatDisp=9.5),

        Component(type="antenna", mass=4.5, dimensions=[.35,.14,.12], heatDisp=9.7),
        Component(type="antenna", mass=3.2, dimensions=[.24,.1,.08], heatDisp=8),
        Component(type="antenna", mass=4.0, dimensions=[.34,.15,.1], heatDisp=9),

        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.3),
        Component(type="star tracker", mass=1.8, dimensions=[.12,.14,.11], heatDisp=1.4),
        Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.1], heatDisp=1.2),

        Component(type="sun sensor", mass=1.2, dimensions=[.1,.09,.08], heatDisp=0.9),
        Component(type="sun sensor", mass=1.1, dimensions=[.1,.08,.07], heatDisp=1),
        Component(type="sun sensor", mass=1.3, dimensions=[.12,.1,.09], heatDisp=1.1),

        Component(type="battery", mass=5.8, dimensions=[.23,.21,.13], heatDisp=2.2),
        Component(type="battery", mass=6.0, dimensions=[.24,.22,.14], heatDisp=2.4),

        Component(type="PCU", mass=7, dimensions=[.26,.2,.14], heatDisp=6.5),
        Component(type="PCU", mass=6.5, dimensions=[.25,.19,.13], heatDisp=6.3),

        Component(type="OBDH", mass=9, dimensions=[.24,.19,.16], heatDisp=5.8),
        Component(type="OBDH", mass=8.8, dimensions=[.23,.18,.15], heatDisp=5.7),

        Component(type="reaction wheel", mass=3, dimensions=[.14,.12,.1], heatDisp=3.2),
        Component(type="reaction wheel", mass=3.2, dimensions=[.15,.13,.11], heatDisp=3.3),

        Component(type="propellant tank", mass=13, dimensions=[.3,.25,.2], heatDisp=4.2),
        Component(type="propellant tank", mass=12.5, dimensions=[.29,.24,.19], heatDisp=4.1),

        Component(type="attitude thruster", mass=2.5, dimensions=[.15,.14,.12], heatDisp=4.8),
        Component(type="attitude thruster", mass=2.7, dimensions=[.16,.15,.13], heatDisp=5),

        Component(type="IMU", mass=2.5, dimensions=[.14,.12,.09], heatDisp=2.5),
        Component(type="IMU", mass=2.4, dimensions=[.13,.11,.08], heatDisp=2.4),

        Component(type="atomic clock", mass=1.8, dimensions=[.12,.11,.07], heatDisp=1.6),

        Component(type="heater", mass=1.2, dimensions=[.09,.07,.05], heatDisp=1.9),
        Component(type="heater", mass=1.3, dimensions=[.1,.08,.06], heatDisp=2),

        Component(type="gyro", mass=3.1, dimensions=[.19,.13,.02], heatDisp=2.8),
        Component(type="gyro", mass=3.0, dimensions=[.18,.12,.02], heatDisp=2.7),

        Component(type="magnetometer", mass=1.4, dimensions=[.1,.12,.1], heatDisp=1.1),
        Component(type="magnetometer", mass=1.5, dimensions=[.11,.13,.09], heatDisp=1.2),

        Component(type="accelerometer", mass=1.8, dimensions=[.13,.11,.09], heatDisp=2.2),
        Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.1)
    ]

    # transferLearningComponents = [
    #     [
    #     Component(type="solar panel", mass=1.8, dimensions=[0.22, 0.53, 0.012], heatDisp=1.7),
    #     Component(type="solar panel", mass=1.5, dimensions=[0.2, 0.48, 0.01], heatDisp=1.5),
    #     Component(type="solar panel", mass=1.7, dimensions=[0.23, 0.52, 0.011], heatDisp=1.6),
    #     Component(type="solar panel", mass=1.6, dimensions=[0.21, 0.49, 0.01], heatDisp=1.4),

    #     Component(type="payload", mass=6.8, dimensions=[0.31, 0.25, 0.22], heatDisp=3.3),
    #     Component(type="payload", mass=5.9, dimensions=[0.29, 0.24, 0.21], heatDisp=3.2),

    #     Component(type="transmitter", mass=4.2, dimensions=[0.26, 0.12, 0.09], heatDisp=12.5),
    #     Component(type="transmitter", mass=3.9, dimensions=[0.25, 0.11, 0.08], heatDisp=11.3),

    #     Component(type="receiver", mass=3.4, dimensions=[0.22, 0.13, 0.06], heatDisp=9.7),
    #     Component(type="receiver", mass=3.7, dimensions=[0.24, 0.14, 0.07], heatDisp=10),

    #     Component(type="antenna", mass=4.2, dimensions=[0.36, 0.15, 0.11], heatDisp=10),
    #     Component(type="antenna", mass=3.8, dimensions=[0.3, 0.12, 0.09], heatDisp=9),
    #     Component(type="antenna", mass=4.5, dimensions=[0.37, 0.16, 0.12], heatDisp=10.2),

    #     Component(type="star tracker", mass=1.9, dimensions=[0.13, 0.15, 0.11], heatDisp=1.5),
    #     Component(type="star tracker", mass=1.6, dimensions=[0.11, 0.13, 0.1], heatDisp=1.3),

    #     Component(type="sun sensor", mass=1.3, dimensions=[0.11, 0.1, 0.08], heatDisp=1.2),
    #     Component(type="sun sensor", mass=1.4, dimensions=[0.12, 0.11, 0.09], heatDisp=1.3),
    #     Component(type="sun sensor", mass=1.2, dimensions=[0.1, 0.09, 0.07], heatDisp=1.1),

    #     Component(type="battery", mass=6.3, dimensions=[0.25, 0.23, 0.15], heatDisp=2.6),
    #     Component(type="battery", mass=6.1, dimensions=[0.24, 0.22, 0.14], heatDisp=2.5),

    #     Component(type="PCU", mass=6.9, dimensions=[0.27, 0.21, 0.15], heatDisp=6.7),
    #     Component(type="PCU", mass=6.6, dimensions=[0.26, 0.2, 0.14], heatDisp=6.4),

    #     Component(type="OBDH", mass=9.5, dimensions=[0.26, 0.2, 0.17], heatDisp=6),
    #     Component(type="OBDH", mass=8.9, dimensions=[0.25, 0.19, 0.16], heatDisp=5.9),

    #     Component(type="reaction wheel", mass=3.4, dimensions=[0.15, 0.13, 0.11], heatDisp=3.5),
    #     Component(type="reaction wheel", mass=3.1, dimensions=[0.14, 0.12, 0.1], heatDisp=3.3),

    #     Component(type="propellant tank", mass=13.2, dimensions=[0.31, 0.26, 0.21], heatDisp=4.3),
    #     Component(type="propellant tank", mass=12.8, dimensions=[0.3, 0.25, 0.2], heatDisp=4.2),

    #     Component(type="attitude thruster", mass=2.6, dimensions=[0.16, 0.15, 0.13], heatDisp=5.2),
    #     Component(type="attitude thruster", mass=2.8, dimensions=[0.17, 0.16, 0.14], heatDisp=5.3),

    #     Component(type="IMU", mass=2.7, dimensions=[0.15, 0.13, 0.1], heatDisp=2.6),
    #     Component(type="IMU", mass=2.5, dimensions=[0.14, 0.12, 0.09], heatDisp=2.5),

    #     Component(type="atomic clock", mass=1.9, dimensions=[0.13, 0.12, 0.08], heatDisp=1.8),
    #     Component(type="atomic clock", mass=2.0, dimensions=[0.14, 0.13, 0.09], heatDisp=1.9),

    #     Component(type="heater", mass=1.4, dimensions=[0.11, 0.09, 0.06], heatDisp=2.1),
    #     Component(type="heater", mass=1.3, dimensions=[0.1, 0.08, 0.05], heatDisp=2),

    #     Component(type="gyro", mass=3.2, dimensions=[0.2, 0.14, 0.03], heatDisp=3),
    #     Component(type="gyro", mass=3.3, dimensions=[0.19, 0.13, 0.03], heatDisp=2.9),

    #     Component(type="magnetometer", mass=1.6, dimensions=[0.11, 0.13, 0.11], heatDisp=1.3),
    #     Component(type="magnetometer", mass=1.5, dimensions=[0.1, 0.12, 0.1], heatDisp=1.2),

    #     Component(type="accelerometer", mass=1.9, dimensions=[0.14, 0.12, 0.1], heatDisp=2.4),
    #     Component(type="accelerometer", mass=1.8, dimensions=[0.13, 0.11, 0.09], heatDisp=2.3),
    #     Component(type="accelerometer", mass=2.0, dimensions=[0.15, 0.13, 0.1], heatDisp=2.5)
    #     ],
    #     [
    #     Component(type="battery", mass=6.2, dimensions=[0.24, 0.22, 0.14], heatDisp=2.5),
    #     Component(type="solar panel", mass=1.7, dimensions=[0.21, 0.51, 0.01], heatDisp=1.6),
    #     Component(type="transmitter", mass=4.3, dimensions=[0.26, 0.11, 0.1], heatDisp=12.6),
    #     Component(type="reaction wheel", mass=3.5, dimensions=[0.16, 0.14, 0.12], heatDisp=3.6),
    #     Component(type="payload", mass=6.7, dimensions=[0.3, 0.24, 0.22], heatDisp=3.4),
        
    #     Component(type="gyro", mass=3.1, dimensions=[0.19, 0.13, 0.03], heatDisp=2.8),
    #     Component(type="sun sensor", mass=1.3, dimensions=[0.11, 0.1, 0.08], heatDisp=1.2),
    #     Component(type="IMU", mass=2.8, dimensions=[0.15, 0.14, 0.1], heatDisp=2.7),
    #     Component(type="OBDH", mass=9.6, dimensions=[0.27, 0.2, 0.16], heatDisp=6.1),
    #     Component(type="PCU", mass=6.8, dimensions=[0.26, 0.21, 0.15], heatDisp=6.6),
        
    #     Component(type="accelerometer", mass=1.9, dimensions=[0.14, 0.12, 0.1], heatDisp=2.3),
    #     Component(type="atomic clock", mass=2.1, dimensions=[0.14, 0.12, 0.1], heatDisp=1.9),
    #     Component(type="magnetometer", mass=1.7, dimensions=[0.11, 0.14, 0.1], heatDisp=1.4),
    #     Component(type="solar panel", mass=1.6, dimensions=[0.22, 0.5, 0.011], heatDisp=1.5),
    #     Component(type="antenna", mass=3.9, dimensions=[0.34, 0.14, 0.1], heatDisp=9.2),
        
    #     Component(type="sun sensor", mass=1.5, dimensions=[0.12, 0.11, 0.09], heatDisp=1.4),
    #     Component(type="receiver", mass=3.8, dimensions=[0.25, 0.14, 0.08], heatDisp=10.2),
    #     Component(type="propellant tank", mass=12.9, dimensions=[0.3, 0.25, 0.2], heatDisp=4.4),
    #     Component(type="payload", mass=6.3, dimensions=[0.28, 0.23, 0.21], heatDisp=3.3),
    #     Component(type="heater", mass=1.4, dimensions=[0.1, 0.08, 0.06], heatDisp=2.0),
        
    #     Component(type="reaction wheel", mass=3.3, dimensions=[0.14, 0.12, 0.11], heatDisp=3.4),
    #     Component(type="OBDH", mass=9.2, dimensions=[0.26, 0.19, 0.17], heatDisp=5.9),
    #     Component(type="IMU", mass=2.5, dimensions=[0.14, 0.12, 0.09], heatDisp=2.5),
    #     Component(type="star tracker", mass=1.8, dimensions=[0.13, 0.15, 0.12], heatDisp=1.5),
    #     Component(type="attitude thruster", mass=2.8, dimensions=[0.16, 0.15, 0.13], heatDisp=5.3),
        
    #     Component(type="solar panel", mass=1.8, dimensions=[0.23, 0.54, 0.012], heatDisp=1.8),
    #     Component(type="antenna", mass=4.0, dimensions=[0.35, 0.15, 0.11], heatDisp=9.8),
    #     Component(type="battery", mass=6.0, dimensions=[0.25, 0.22, 0.15], heatDisp=2.7),
    #     Component(type="atomic clock", mass=2.0, dimensions=[0.13, 0.12, 0.09], heatDisp=1.8),
    #     Component(type="magnetometer", mass=1.6, dimensions=[0.1, 0.13, 0.11], heatDisp=1.3),
        
    #     Component(type="gyro", mass=3.0, dimensions=[0.18, 0.12, 0.02], heatDisp=2.7),
    #     Component(type="accelerometer", mass=2.0, dimensions=[0.15, 0.13, 0.11], heatDisp=2.5),
    #     Component(type="heater", mass=1.5, dimensions=[0.11, 0.09, 0.07], heatDisp=2.1),
    #     Component(type="PCU", mass=6.7, dimensions=[0.25, 0.2, 0.14], heatDisp=6.4),
    #     Component(type="receiver", mass=3.6, dimensions=[0.22, 0.13, 0.07], heatDisp=9.6),
        
    #     Component(type="attitude thruster", mass=2.6, dimensions=[0.15, 0.14, 0.12], heatDisp=5.0),
    #     Component(type="antenna", mass=4.3, dimensions=[0.36, 0.16, 0.12], heatDisp=10.1),
    #     Component(type="reaction wheel", mass=3.4, dimensions=[0.15, 0.13, 0.11], heatDisp=3.5),
    #     Component(type="propellant tank", mass=13.1, dimensions=[0.31, 0.26, 0.21], heatDisp=4.5),
    #     Component(type="star tracker", mass=1.9, dimensions=[0.14, 0.16, 0.13], heatDisp=1.6),
        
    #     Component(type="transmitter", mass=4.1, dimensions=[0.25, 0.12, 0.09], heatDisp=12.4),
    #     Component(type="solar panel", mass=1.9, dimensions=[0.24, 0.55, 0.013], heatDisp=1.9),
    #     Component(type="payload", mass=6.5, dimensions=[0.29, 0.24, 0.22], heatDisp=3.5),
    #     Component(type="gyro", mass=3.2, dimensions=[0.2, 0.13, 0.02], heatDisp=2.9),
    #     Component(type="sun sensor", mass=1.2, dimensions=[0.1, 0.09, 0.07], heatDisp=1.1)
    #     ],
    #     [
    #     Component(type="transmitter", mass=4.2, dimensions=[0.27, 0.11, 0.1], heatDisp=12.7),
    #     Component(type="reaction wheel", mass=3.8, dimensions=[0.15, 0.14, 0.12], heatDisp=3.9),
    #     Component(type="solar panel", mass=1.9, dimensions=[0.22, 0.52, 0.012], heatDisp=1.8),
    #     Component(type="OBDH", mass=9.5, dimensions=[0.26, 0.19, 0.16], heatDisp=6.3),
    #     Component(type="gyro", mass=3.4, dimensions=[0.18, 0.12, 0.02], heatDisp=2.6),

    #     Component(type="payload", mass=6.9, dimensions=[0.31, 0.25, 0.23], heatDisp=3.6),
    #     Component(type="antenna", mass=4.5, dimensions=[0.37, 0.15, 0.1], heatDisp=9.7),
    #     Component(type="IMU", mass=3.1, dimensions=[0.15, 0.13, 0.1], heatDisp=2.9),
    #     Component(type="sun sensor", mass=1.4, dimensions=[0.12, 0.11, 0.09], heatDisp=1.3),
    #     Component(type="star tracker", mass=1.8, dimensions=[0.13, 0.15, 0.11], heatDisp=1.5),

    #     Component(type="receiver", mass=4.0, dimensions=[0.24, 0.14, 0.08], heatDisp=10.5),
    #     Component(type="accelerometer", mass=2.1, dimensions=[0.16, 0.13, 0.11], heatDisp=2.4),
    #     Component(type="magnetometer", mass=1.7, dimensions=[0.11, 0.13, 0.09], heatDisp=1.4),
    #     Component(type="heater", mass=1.6, dimensions=[0.12, 0.1, 0.08], heatDisp=2.2),
    #     Component(type="atomic clock", mass=2.2, dimensions=[0.15, 0.13, 0.1], heatDisp=1.7),

    #     Component(type="PCU", mass=6.9, dimensions=[0.27, 0.21, 0.16], heatDisp=6.5),
    #     Component(type="battery", mass=6.4, dimensions=[0.26, 0.23, 0.15], heatDisp=2.8),
    #     Component(type="solar panel", mass=1.8, dimensions=[0.23, 0.53, 0.013], heatDisp=1.9),
    #     Component(type="transmitter", mass=4.0, dimensions=[0.25, 0.12, 0.09], heatDisp=12.3),
    #     Component(type="attitude thruster", mass=2.9, dimensions=[0.16, 0.14, 0.12], heatDisp=5.4),

    #     Component(type="propellant tank", mass=13.2, dimensions=[0.32, 0.26, 0.2], heatDisp=4.3),
    #     Component(type="reaction wheel", mass=3.7, dimensions=[0.16, 0.13, 0.1], heatDisp=3.5),
    #     Component(type="gyro", mass=3.0, dimensions=[0.2, 0.13, 0.03], heatDisp=2.8),
    #     Component(type="payload", mass=6.6, dimensions=[0.3, 0.25, 0.22], heatDisp=3.3),
    #     Component(type="sun sensor", mass=1.5, dimensions=[0.11, 0.1, 0.09], heatDisp=1.4)
    #     ],
    #     [
    #     Component(type="solar panel", mass=1.7, dimensions=[0.21, 0.51, 0.01], heatDisp=1.5),
    #     Component(type="battery", mass=6.3, dimensions=[0.24, 0.22, 0.14], heatDisp=2.6),
    #     Component(type="payload", mass=6.4, dimensions=[0.28, 0.23, 0.21], heatDisp=3.5),
    #     Component(type="OBDH", mass=9.3, dimensions=[0.26, 0.19, 0.17], heatDisp=6.0),
    #     Component(type="sun sensor", mass=1.2, dimensions=[0.1, 0.09, 0.07], heatDisp=1.1),

    #     Component(type="antenna", mass=3.9, dimensions=[0.35, 0.15, 0.12], heatDisp=9.3),
    #     Component(type="reaction wheel", mass=3.6, dimensions=[0.15, 0.13, 0.11], heatDisp=3.4),
    #     Component(type="atomic clock", mass=2.1, dimensions=[0.14, 0.13, 0.09], heatDisp=1.8),
    #     Component(type="magnetometer", mass=1.5, dimensions=[0.11, 0.12, 0.09], heatDisp=1.3),
    #     Component(type="accelerometer", mass=2.0, dimensions=[0.15, 0.12, 0.1], heatDisp=2.4),

    #     Component(type="receiver", mass=3.7, dimensions=[0.23, 0.14, 0.08], heatDisp=10.1),
    #     Component(type="gyro", mass=3.2, dimensions=[0.19, 0.12, 0.02], heatDisp=2.9),
    #     Component(type="IMU", mass=3.0, dimensions=[0.14, 0.13, 0.1], heatDisp=2.6),
    #     Component(type="PCU", mass=6.8, dimensions=[0.26, 0.21, 0.15], heatDisp=6.7),
    #     Component(type="transmitter", mass=4.1, dimensions=[0.25, 0.13, 0.1], heatDisp=12.2),

    #     Component(type="propellant tank", mass=12.8, dimensions=[0.3, 0.24, 0.19], heatDisp=4.1),
    #     Component(type="star tracker", mass=1.9, dimensions=[0.14, 0.16, 0.12], heatDisp=1.6),
    #     Component(type="heater", mass=1.6, dimensions=[0.12, 0.1, 0.08], heatDisp=2.1),
    #     Component(type="attitude thruster", mass=2.7, dimensions=[0.15, 0.14, 0.11], heatDisp=5.3),
    #     Component(type="reaction wheel", mass=3.5, dimensions=[0.14, 0.12, 0.1], heatDisp=3.2)
    #     ]
    # ]

    transferLearningComponents = None

    return componentList, transferLearningComponents