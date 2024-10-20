from py4j.java_gateway import JavaGateway

archPath = r"C:\Users\demagall\Documents\VS Code\Research\SpaDes\arch.json"
gateway = JavaGateway()
testApp = gateway.entry_point
result = testApp.getArchitectureScience(archPath)
print(result)