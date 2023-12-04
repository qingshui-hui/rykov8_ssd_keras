
cd .devcontainer
docker-compose build 
docker login
docker push saiki987/keras_ssd:1.15
