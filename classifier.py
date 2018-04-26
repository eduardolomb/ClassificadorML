import turicreate as turi
# coding=<utf8>

url = "dataset/"

data = turi.image_analysis.load_images(url)

 
labels = ['android', 'arvore', 'cadeira', 'calculadora','camiseta','caneca','carro', 'casa', 'celular', 'comida', 'computador', 'copo', 'floresta','guarda-chuva', 'iPhone', 'mesa', 'monitor', 'moto', 'mouse', 'paisagem', 'peixe','pessoa', 'piscina','pneu', 'praia','rosto', 'televisao']  
      
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
