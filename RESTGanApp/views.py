from django.http.response import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from io import BytesIO
from RESTGanApp.genHeightMap import genHeightMap
import base64
import io


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