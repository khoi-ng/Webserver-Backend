from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path('api/', RESTGanApp.site.urls)
      path('api-auth/', include('rest_framework.urls')),
      path('generateHeightMap/', views.GenHeightMap.as_view(), name='Generate Heightmap'),
      path('getHeightMap/', views.GetGanHeightMap.as_view(), name='GetHeightmap'),
]
