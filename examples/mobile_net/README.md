# TFMin Mobile Net Example Project

Adapted from the Tensorflow mobile net implementation published by [Mostafa Gamal Badawy](https://github.com/MG2033) in
his github repository [MG2033/MobileNet](https://github.com/MG2033/MobileNet).

### Description from original repository
An implementation of `Google MobileNet` introduced in TensorFlow. According to the authors, `MobileNet` is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It can be used for different applications including: Object-Detection, Finegrain Classification, Face Attributes and Large Scale Geo-Localization.

Link to the original paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

This implementation was made to be clearer than TensorFlow original implementation. It was also made to be an example of a common DL software architecture. The weights/biases/parameters from the pretrained ImageNet model that was implemented by TensorFlow are dumped to a dictionary in pickle format file (`pretrained_weights/mobilenet_v1.pkl`) to allow a less restrictive way of loading them.



## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

