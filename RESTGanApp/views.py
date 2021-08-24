from django.http.response import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from django.http import FileResponse
import base64
from io import BytesIO
import io

def genHeightMap():
    num_img = 1
    noise_dim = 256

    generator = keras.models.load_model("Generator.h5")
    random_latent_vectors = tf.random.normal(shape=(num_img, noise_dim))

    generated_images = generator.predict(random_latent_vectors)
    generated_images = (generated_images * 127.5) + 127.5
    img = generated_images[0]

    
    img = keras.preprocessing.image.array_to_img(img)
    # img.save("output\\generated_img_{i}.png".format(i=1))
    return img


class GenHeightMap(APIView):
    def get(self, request, *args, **kwargs):
        data = genHeightMap()
        if data is None:
            print("data is nonetype")
        else:
            print("type is: ", type(data))
            print(data)
        img_byte_arr = io.BytesIO()
        data.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return HttpResponse(img_byte_arr, content_type="image/png")



class GetGanHeightMap(APIView):
    def get(self, request, *args, **kwargs):
        data = genHeightMap()
        buffered = BytesIO()
        data.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        res = {
            'image': img_str,
        }
        return Response(res)