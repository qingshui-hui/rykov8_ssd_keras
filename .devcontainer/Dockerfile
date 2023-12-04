FROM tensorflow/tensorflow:1.15.5-jupyter

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev git

RUN pip install scipy==1.1.0 imageio lxml keras==1.2.2 opencv-python==3.4.18.65 
