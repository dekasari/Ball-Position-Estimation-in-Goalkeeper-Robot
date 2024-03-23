# Ball-Position-Estimation-in-Goalkeeper-Robot

### Introduction
This project presents a solution for accurately estimating the position of a ball in flight relative to a goalkeeper robot. By leveraging an Intel RealSense Depth Camera D455 and employing advanced computer vision techniques, this endeavor offers precise real-time calculations crucial for effective robotic goalkeeping.

### Purpose
The goal of this project is to equip a goalkeeper robot with the capability to anticipate and intercept incoming balls by accurately estimating their position in three-dimensional space. This is achieved through a combination of depth sensing technology from the Intel RealSense Depth Camera D455, sophisticated object recognition algorithms, and TensorFlow-based training data preparation.

### Tech Stacks
1. **Intel RealSense Depth Camera D455**: The system relies on the advanced capabilities of the Intel RealSense Depth Camera D455, positioned above the robot's head. This depth camera provides crucial depth information essential for precise localization of the ball in 3D space.
2. **Intel RealSense SDK**: The Intel RealSense SDK is utilized to interface with the Intel RealSense Depth Camera D455, enabling seamless integration and access to its depth sensing capabilities.
3. **OpenCV DNN Library**: For ball detection, a deep learning model based on MobileNet SSD (Single Shot MultiBox Detector) is employed. This model is implemented using the OpenCV DNN (Deep Neural Network) library, enabling robust and efficient detection of the ball within the camera's field of view.
4. **TensorFlow**: TensorFlow is used for training data preparation. This enables the creation of custom object detection models tailored to the specific requirements of the project. See [How to Train an Object Detection Neural Network Using Tensorflow](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10).

### Key Features
- **Real-time Ball Detection**: This project provides real-time detection of the ball's position, allowing the goalkeeper robot to react swiftly to incoming threats.
- **Position Estimation**: By leveraging the depth information provided by the Intel RealSense Depth Camera D455 and the MobileNet SSD deep learning model, this project offers precise estimation of the ball's position in three-dimensional space.

