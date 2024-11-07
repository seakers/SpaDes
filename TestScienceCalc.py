<<<<<<< HEAD
from py4j.java_gateway import JavaGateway

archPath = r"C:\Users\demagall\Documents\VS Code\Research\SpaDes\adjArch.json"
gateway = JavaGateway()
testApp = gateway.entry_point
result = testApp.getArchitectureScience(archPath)
print(result)
=======
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
>>>>>>> a35e2d279df1947dcd7a53eafdc08af8e835c808
