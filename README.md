# Neural-Vision Repository

This repository contains multiple projects focused on different Computer Vision applications, starting with facial recognition and object detection, and culminating in the implementation of a Deep Convolutional GAN (DCGAN) trained on the CIFAR-10 dataset. Each project is self-contained and demonstrates key techniques in computer vision and generative modeling.

## Projects

### 1. Facial Recognition Project
#### Overview
This project focuses on identifying and labeling human faces in images using pre-trained models and advanced image processing techniques.

#### Features
- Implementation using pre-trained facial recognition models.
- Integrated pipelines for dataset creation and training.
- Real-time face detection with OpenCV.

#### Scaling Up Potential
- Enhance performance with larger datasets.
- Extend to real-time applications in security, attendance systems, and user authentication.

#### How to Run
1. Navigate to the facial recognition directory:
   ```bash
   cd facial_recognition
   ```
2. Run the script:
   ```bash
   python facial_recognition.py
   ```
3. Results, including labeled images, will be saved in the `results` directory.

---

### 2. Object Detection Project
#### Overview
This project utilizes state-of-the-art object detection models like YOLOv5 and Single Shot Detectors (SSDs) to identify objects in images and videos.

#### Features
- Support for pre-trained models and custom fine-tuning.
- Integration with OpenCV for preprocessing and visualization.
- Real-time object detection capabilities.

#### Scaling Up Potential
- Fine-tune models on custom datasets for specific use cases.
- Expand to multi-object tracking and predictive analytics.

#### How to Run
1. Navigate to the object detection directory:
   ```bash
   cd object_detection
   ```
2. Run the script:
   ```bash
   python object_detection.py
   ```
3. Detected objects and labeled outputs will be saved in the `results` directory.

---

### 3. Deep Convolutional GANs (DCGAN) Project
#### Overview
Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow in 2014 and have since become a cornerstone of generative modeling. This project implements a DCGAN to generate realistic images from random noise vectors.

#### Features
- **Dataset**: CIFAR-10, a collection of 60,000 32x32 color images in 10 classes.
- **Generator**: A deep convolutional neural network that takes random noise as input and outputs 64x64 images.
- **Discriminator**: A deep convolutional neural network that distinguishes real images from generated ones.
- **Training Loop**: Alternates between training the discriminator and generator.
- **Image Saving**: Saves real and generated samples during training for visualization.

#### Scaling Up Potential
- Train on higher resolution datasets by modifying network architectures.
- Extend to Conditional GANs (cGANs) for class-specific image generation.
- Apply to applications like video generation, style transfer, and more.

#### How to Run
1. Navigate to the DCGAN directory:
   ```bash
   cd dcgan
   ```
2. Run the training script:
   ```bash
   python GAN.py
   ```
3. Generated images will be saved in the `results` directory.

---

## Requirements
- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- OpenCV
- YOLOv5 dependencies (e.g., PyYAML, torchmetrics)

Install the required libraries:
```bash
pip install torch torchvision numpy matplotlib opencv-python
pip install -r yolov5/requirements.txt
```

---

## Repository Structure
```
Neural-Vision/
|— facial_recognition/
|   |— facial_recognition.py
|   |— results/
|— object_detection/
|   |— object_detection.py
|   |— results/
|— dcgan/
|   |— GAN.py
|   |— results/
|— data/
|— README.md
```

---

## Acknowledgments
- **Facial Recognition and Object Detection**: Inspired by advancements in object detection by Joseph Redmon (YOLO) and SSD research.
- **DCGANs**: Based on the concepts introduced by Ian Goodfellow in *"Generative Adversarial Nets"* (2014) and the DCGAN paper by Alec Radford, Luke Metz, and Soumith Chintala.

---

## Visuals
Sample outputs for each project are saved in their respective `results` directories. Highlights include:
- **Facial Recognition**: Labeled images with detected faces.
- **Object Detection**: Images with bounding boxes for detected objects.
- **DCGAN**: Generated CIFAR-10-like images at 64x64 resolution.

---

## Call to Action
Contributions and feedback are welcome! Whether you're exploring computer vision or generative modeling, feel free to clone the repository, test the code, and share your insights. Let's build Neural-Vision together!

---

## License
This project is open-source and available under the MIT License. See the LICENSE file for details.



