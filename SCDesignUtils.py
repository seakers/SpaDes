import numpy as np

def chooseComponent(req,reqName,compInstance,data,obj):
    """
    Chooses the lowest mass component that fulfills the requirement
    """
    choicesReq = []
    compChoices = []
    choicesMass = []
    compTypes = []
    newComp = compInstance
    mode = "closest" # if a component fits the requirement, switch mode to "normal"
    
    for compType in data:

        reqArr = data[compType][reqName].to_numpy()
        massArr = data[compType]["Mass (kg)"].to_numpy()
        if obj == "less":
            if ~(~(reqArr <= req)).all():
                choiceMass = massArr[reqArr <= req].min()
                choiceInd = np.where(massArr == choiceMass)[0][0]
                mode = "normal"
            else:
                choiceInd = reqArr.argmin()
                choiceMass = massArr[choiceInd]
        elif obj == "greater":
            if ~(~(reqArr >= req)).all():
                choiceMass = massArr[reqArr >= req].min()
                choiceInd = np.where(massArr == choiceMass)[0][0]
                mode = "normal"
            else:
                choiceInd = reqArr.argmax()
                choiceMass = massArr[choiceInd]

        choicesReq.append(reqArr[choiceInd])
        choicesMass.append(choiceMass)
        compChoices.append(choiceInd)
        compTypes.append(compType)

    if mode == "normal":
        choicesMassArr = np.asarray(choicesMass)
        finalCompChoice = choicesMassArr.argmin()
    elif mode == "closest":
        choicesReqArr = np.asarray(choicesReq)
        if obj == "greater":
            finalCompChoice = choicesReqArr.argmax()
        elif obj == "less":
            finalCompChoice = choicesReqArr.argmin()
    chosenComponent = data[compTypes[finalCompChoice]].iloc[compChoices[finalCompChoice]]

    newComp.type = compTypes[finalCompChoice]
    newComp.mass = chosenComponent['Mass (kg)']
    newComp.dimensions = [chosenComponent['Length (m)'],chosenComponent['Width (m)'],chosenComponent['Height (m)']]
    newComp.avgPower = chosenComponent['Avg Power (W)']
    newComp.peakPower = chosenComponent['Peak Power (W)']
    newComp.name = chosenComponent['Name']
    newComp.tempRange = [chosenComponent['Temp Min (C)'],chosenComponent['Temp Max (C)']]
    newComp.performance = chosenComponent[reqName]
    
    return newComp