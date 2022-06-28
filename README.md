# COVID-19-Diagnosis-from-Chest-X-Rays-using-Machine-Learning-Bachelor-Project-
The new coronavirus 2019 (COVID-2019) was first detected in Wuhan, China, in December 2019 and quickly spread over the world, becoming a pandemic. It has had disastrous consequences for people’s daily life, public health, and the global economy. As a result, there is an urgent need for rapid identification with clear visualization that could save a suspected COVID-19 patient. According to recent studies gained using radiological imaging techniques, such images convey important information about the COVID-19 virus. Advanced artificial intelligence (AI) techniques combined with radiological imaging can aid in the accurate detection of this disease, as well as help overcome the problem of a lack of specialized physicians in remote villages. They can also provide more promising results than RT-PCR testing, allowing for more accurate detection and prediction of COVID-19 cases. In this thesis, we have presented more than one model to efficiently classify COVID-19 infected patients or normal or pneumonia using chest X-rays. 
Deep learning models such as Convolutional Neural Networks (CNNs) and using pretrained models especially DenseNet-121 showed remarkable results especially the DenseNet-121 with an accuracy of 100% for binary classification and 99.77% for multiclass classification. This model would be extremely useful in this pandemic, when the disease burden and the need for preventive measures are at odds with the currently available resources, and they would also provide much-needed assistance to healthcare professionals in order to prevent the virus from spreading.


Here is an example of the simple interface that we created.
-----------------------------

![GUI](https://user-images.githubusercontent.com/83249350/176219404-c0dadf09-a7ac-4539-b2f6-40b5b6063ab8.PNG)

![GUI Pneumonia](https://user-images.githubusercontent.com/83249350/176219451-f6d77c29-cecd-4b39-8688-73d6a841831a.PNG)

![GUI Normal](https://user-images.githubusercontent.com/83249350/176219464-9643dacd-59cd-430c-9ccc-445190444617.PNG)

Here is the Confision matrix of our model

![2 Classes DenNet 99 8](https://user-images.githubusercontent.com/83249350/176219838-7fa9419e-e995-4e06-a405-c6027624e027.png)
![3 Classes Modified New](https://user-images.githubusercontent.com/83249350/176219853-09421ffd-e3d5-406e-a4a9-89b764444d31.png)


# Inorder to use our System, you need to have anaconda downloaded and installed
in the anaconda command prompt type
- cd to the file directory
- set FLASK_APP=Flask.py
- flask run --host=0.0.0.0
