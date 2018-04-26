import turicreate as turi
# coding=<utf8>

url = "dataset/"

data = turi.image_analysis.load_images(url)

#data["imageType"] = data["path"].apply(lambda path: "Android" if "android" in path else "Arvore" if "arvore" in path else "Cadeira" if "cadeira" in path else "Casa" if "casa" in path else "Celular" if "celular" in path else "Computador" if "computador" in path else "Copo" if "copo" in path else "Floresta" if "floresta" in path else "iPhone" if "iPhone" in path else "Mesa" if "mesa" in path else "Monitor" if "monitor" in path else "Mouse" if "mouse" in path else "Paisagem" if "paisagem" in path else "Piscina" if "piscina" in path else "Praia" if "praia" in path else "Televisao" if "televisao" in path else "Comida" if "comida" in path else "Carro" if "carro" in path else "Moto" if "moto" in path else "Pessoa") 
 
labels = ['android', 'arvore', 'cadeira', 'carro', 'casa', 'celular', 'comida', 'computador', 'copo', 'floresta', 'iPhone', 'mesa', 'monitor', 'moto', 'mouse', 'paisagem', 'pessoa', 'piscina', 'praia', 'televisao']  
      
def get_label(path, labels=labels):  
    for label in labels:  
        if label in path:  
            return label  
  
data['label'] = data['path'].apply(get_label)  
 

data.save("imageType.sframe")  
dataBuffer = turi.SFrame("imageType.sframe")
 
# Make a train-test split  
train_data, test_data = dataBuffer.random_split(0.9)  
 
# Create a model  
model = turi.image_classifier.create(train_data, target='label', model="resnet-50")  
 
# Save predictions to an SFrame (class and corresponding class-probabilities)  
predictions = model.classify(test_data)  
 
# Evaluate the model and save the results into a dictionary  
results = model.evaluate(test_data)  
print "Accuracy         : %s" % results['accuracy']  
print "Confusion Matrix : \n%s" % results['confusion_matrix']  
 
# Save the model for later usage in Turi Create  
model.save('imageType.model')  

model.export_coreml("imageClassifier.mlmodel")
