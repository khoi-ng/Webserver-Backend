# Webserver-Backend (Django Rest API)

The Backend is using a WGAN Keras-Model to generate Heightmaps.

A 3D-Frontend for the Backend can be found [here](https://github.com/ibrahimSchechsaher/Webserver-Frontend).

The GAN where the model was created can be found [here](https://github.com/Erik3003/terrain_gan)


## How to start the Backend-Server

```bash
# start the server
python manage.py runserver

# Here you can generate Random Heightmaps visually:
http://127.0.0.1:8000/generateHeightMap/

# This is the API endpoint, which serves a random Heightmap as 64-base encoded String
http://127.0.0.1:8000/getHeightMap/

```
