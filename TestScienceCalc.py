from py4j.java_gateway import JavaGateway
import os

def testScienceCalc(archPath, revisit):
    # archPath = r"C:\Users\demagall\Documents\VS Code\Research\SpaDes\adjArch.json"
    archPath = os.path.abspath(archPath)
    gateway = JavaGateway()
    testApp = gateway.entry_point
    science = testApp.getArchitectureScience(archPath, revisit)
    # print(result)
    return science
