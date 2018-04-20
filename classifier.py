import turicreate as turi
# coding=<utf8>

url = "dataset/"

data = turi.image_analysis.load_images(url)

data["imageType"] = data["path"].apply(lambda path: "Android" if "android" in path else "Arvore" if "arvore" in path else "Cadeira" if "cadeira" in path else "Casa" if "casa" in path else "Celular" if "celular" in path else "Computador" if "computador" in path else "Copo" if "copo" in path else "Floresta" if "floresta" in path else "iPhone" if "iPhone" in path else "Mesa" if "mesa" in path else "Monitor" if "monitor" in path else "Mouse" if "mouse" in path else "Paisagem" if "paisagem" in path else "Piscina" if "piscina" in path else "Praia" if "praia" in path else "Televisao" if "televisao" in path else "Comida" if "comida" in path else "Carro" if "carro" in path else "Moto") 




data.save("imageType.sframe")

data.explore()


dataBuffer = turi.SFrame("imageType.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)
model = turi.image_classifier.create(trainingBuffers, target="imageType", model="resnet-50")
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("imageType.model")
model.export_coreml("imageClassifier.mlmodel")
