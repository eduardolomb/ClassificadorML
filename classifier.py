import turicreate as turi

url = "dataset/"

data = turi.image_analysis.load_images(url)

data["path"].apply(lambda path: "Arvore" if "arvore" in path)
data["path"].apply(lambda path: "Casa" if "casa" in path)
data["path"].apply(lambda path: "Celular" if "celular" in path)
data["path"].apply(lambda path: "Computador" if "computador" in path)
data["path"].apply(lambda path: "Copo" if "copo" in path)
data["path"].apply(lambda path: "Floresta" if "floresta" in path)
data["path"].apply(lambda path: "Monitor" if "monitor" in path)
data["path"].apply(lambda path: "Mouse" if "mouse" in path)
data["path"].apply(lambda path: "Piscina" if "piscina" in path)

data["imageType"] = data["path"]

data.save("imageType.sframe")

data.explore()


dataBuffer = turi.SFrame("imageType.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)
model = turi.image_classifier.create(trainingBuffers, target="imageType", model="resnet-50")
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("imageType.model")
model.export_coreml("imageClassifier.mlmodel")