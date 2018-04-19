import turicreate as turi

url = "dataset/"

data = turi.image_analysis.load_images(url)

data["imageType"] = data["path"].apply(lambda path: "Arvore" if "arvore" in path else "Casa" if "casa" in path else "Celular" if "celular" in path else "Computador" if "computador" in path else "Copo" if "copo" in path else "Floresta" if "floresta" in path else "Monitor" if "monitor" in path else "Mouse" if "mouse" in path else "Piscina")




data.save("imageType.sframe")

data.explore()


dataBuffer = turi.SFrame("imageType.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)
model = turi.image_classifier.create(trainingBuffers, target="imageType", model="resnet-50")
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("imageType.model")
model.export_coreml("imageClassifier.mlmodel")
