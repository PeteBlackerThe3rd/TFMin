# TFMin - Minimal TensorFlow

A library to export light weight ansi-c implementations of TensorFlow graphs, for use on
micro-processor and embedded systems.

## Overview

The TFMin library allows you to convert a TensorFlow graph within a python script into an ansi-c
implementation with no library dependencies. This allows the generated c code
to be compiled on small computer systems and embbedded systems. Unlike the standard c++
implementation of TensorFlow which is already available the binaries produced by TFMin
do not have dependencies on large shared object libraries. These dependencies can make 
implementing this code on embedded systems, difficult or impossible.

This package is a python tool that is used to analyse and export
the flow graphs to ansi-c code. An example is provided where an MNIST classifier
is trained using TensorFlow python then exported to c++ and built into a native binary.
Tensorflow lite flatbuffers are also directly supported meaning a wide range of pre-built
models can easily be imported and deployed.

This open source software has been developed during my PhD research at the Surrey Space 
Centre, at Surrey University and made possible by support from Airbus.

![](logos/airbus-ds-logo.png)
![](logos/ssc-logo.png)
![](logos/university-of-surrey-logo.png)

## Installation Instructions

Clone this repository and add the required locations to your PYTHONPATH.

## Licence

This software is (c) 2019 Pete Blacker, Surrey Space Centre & Airbus Defence and Space Ltd. It is licenced under the GPL v3 license for more details see the LICENCE file.

## Airbus Commercial TFMin - LEON3 Support

A commercial extenstion to the TFMin tool is available from Airbus, which has full support for
the LEON family of radiation hardened processors. This includes tensor operations  optimised 
for these processors and V&V assurance to industry standards. For further information about 
features and licencing please contact: ilke.karsli@airbus.com, arthur.scharf@airbus.com or carlos.hervasgarcia@airbus.com

## Publications

If you use this software for research purposes please cite the following publication in any
published work.

* [Blacker, P., Bridges, C.P., Hadfield, S., 2019, July. Rapid Prototyping of Deep Learning Models on
Radiation Hardened CPUs. In Thirteenth NASA/ESA Conference on Adaptive Hardware and
Systems (AHS 2019). IEEE.](http://epubs.surrey.ac.uk/852310/1/Rapid%20Prototyping%20of%20Deep%20Learning%20Models%20on%20Radiation%20Hardened%20CPUs.pdf)

## Developer Guides

The attached wiki contains an installation guide and a set of tutorials to introduce 
new developers to TFMin and walks through how to export and build C++ implementations
of existing TensorFlow models.
