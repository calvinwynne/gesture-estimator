# Gesture Estimator



## Gesture estimation using Deeplearning and Mediapipe


![alt text](https://github.com/calvinwynne/gesture-estimator/blob/main/videos/video.gif?raw=true)

  The primary purpose of this research is to devise a methodology that allows for teaching various Neural Network models to distinguish between different gestures with limited data. This is made possible using Google’s MediaPipe framework, to estimate the positions of vital landmarks on a human body that are responsible for movement. Based on these extracted landmark positions, an application is developed for this research, to extract multiple useful features that can be used to train three different types of NN models namely: Deep Neural Network, Convolution Neural Network, and Long Short-Term Memory Neural Network model and to evaluate their performance. To achieve this, a custom video dataset of 11 different actions and a sample size of 45 each, were collected. These models are then trained to identify a sequence of actions, that includes both explicit and implicit gestures. 

  The key motivation behind this research is to develop and deploy a gesture detection model that detects both implicit and explicit actions, is lightweight, efficient, and performant enough to be deployed on UAVs to execute real-time gesture detection with little to no latency, and to decide based on the gesture being detected. Finally, to evaluate the performance of these three chosen models on various performance metrics such as classification accuracy, execution time, training time, learning rate over a fixed number of epochs, overall model performance and to ultimately test the accuracy over a video clip that is captured with an entirely different subject in a different environment.
