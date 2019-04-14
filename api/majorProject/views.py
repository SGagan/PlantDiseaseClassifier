import datetime
import pickle
import json
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from api.settings import BASE_DIR
import io
import base64
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array


from custom_code import image_converter

@api_view(['GET'])
def __index__function(request):
    start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_time_ms = (elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data = {
        "error" : "0",
        "message" : "Successful",
        "restime" : elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')

@api_view(['POST','GET'])
def predict_plant_disease(request):
    try:
        if request.method == "GET" :
            return_data = {
                "error" : "0",
                "message" : "Plant Disease Classifier Api"
            }
        else:
            if request.body:
                request_data = request.data["plant_image"]
                # header, image_data = request_data.split(';base64,')
                # image_array, err_msg = image_converter.convert_image(image_data)
                image_array = image_converter.convert_image_to_array(request_data)
                image_array = np.array(image_array, dtype=np.float16) / 225.0
                image_array = np.expand_dims(image_array, axis=0)
                # if err_msg == None :


                # Enter your model path 
                model_file = f"{BASE_DIR}/ml_files/cnn_model(2).pkl"
                saved_classifier_model = pickle.load(open(model_file,'rb'))
                prediction = saved_classifier_model.predict(image_array)

                # Enter your label_transform path
                label_binarizer = pickle.load(open(f"{BASE_DIR}/ml_files/label_transform2.pkl",'rb'))
                result_rate = prediction[0].tolist()
                total_class = (label_binarizer.classes_).tolist()
                    # construct the result
                result_class = dict()
                for i in range(len(total_class)):
                    result_class[total_class[i]] = result_rate[i]
                result_class = sorted(result_class.items(), key=lambda x:x[1], reverse=True)
                print(np.array(result_class))
                return_data = {
                    "error" : "0",
                    "data" : f"{result_class[0]}"
                    }
                # else :
                #     return_data = {
                #         "error" : "4",
                #         "message" : f"Error : {err_msg}"
                #     }
            else :
                return_data = {
                    "error" : "1",
                    "message" : "Request Body is empty",
                }
    except Exception as e:
        return_data = {
            "error" : "3",
            "message" : f"Error : {str(e)}",
        }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')