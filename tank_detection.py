from imageai.Detection import ObjectDetection
from imageai.Prediction.Custom import ModelTraining
import os

execution_path = os.getcwd()

detector = ObjectDetection()



model_trainer = ModelTraining()
model_trainer.setModelTypeAsInceptionV3()
model_trainer.setDataDirectory('dataset')
model_trainer.trainModel(num_objects=5, num_experiments=2, enhance_data=False, batch_size=32, show_network_summary=True)





'''
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "model_tanks"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "st.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
'''