# Gravitational_wave_detection
The main objective of this project is to make machine learning model that helps us to classify Gravitational wave(GW) signal from noise signal detected by LIGO observatories.

**Introduction:**

Gravitational waves are considered to be the signals from Colliding blackholes, the GW signals helps researchers to understand more concepts about outer space like Neutron star mergers and black hole properties. These signals are tiny ripples in fabric of space-time. These were detected by GW worldwide detectors. But this signals were buried under detector noise. So,our target is to identify whether any GW signal is present in the given signal along with detector noise.



**Data Collection:**

We are provided with a training set of time series data containing simulated gravitational wave measurements from 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo). The waves detected by GW detectors will have noises in output signals. So our aim is to find out if the output signal from this detector is only noise or signal+noise.
Target values: → 0 means negative sample (no gravitational wave)
  		         → 1 means positive sample (gravitational wave present)
There is a total of 560000 npy files of train data. Each npy file contains 3-time series (1 for each detector) and each spans 2 sec and is sampled at 2,048 Hz

**Data Preprocessing**

To check whether a  GW signal is present or not along with the noise, we need to check if there is any other frequency spread in the signal along with noise frequency. But here the main problem is, we are given the signal data in Time domain, we need to convert it into frequency domain (i’e; into spectrogram) which are stored as images. These spectrogram images are later divided into train and validation datasets of several batches. We use Constant Q Transform technique to convert the signals in time domain to spectrogram.

**CNN Model**

Among all the neural networks present, CNN is mostly used and best suitable classification model for image classification. Images generally are of high dimensions, and when we try deal with data with high dimensions or large number of parameters, ANN’s will fail to provide better accuracy. And we can also use LSTM networks when the considered data is in time series domain. But for the ease of our problem we are converting in into frequency domain, so CNN is best suitable one we are left with.

**Results**

After creating the Sequential CNN model, we shall train the model and find the accuracy of the predictions made on validation dataset. The model we created uses Adam optimizer with learning rate of 0.0001, and loss function of the model is binary cross entropy. By changing any one of this 3 parameters, the final resultant accuracy of the model also changes.
The accuracy we got with the above mentioned parameters is:
Accuracy of the model: 0.8343 
Precision_score: 0.8271 
Recall_score: 0.8428 
F1_score: 0.8283 

The accuracy we got here is not constant. It varies accordingly with changing the parameters of the model. Like changing learning rate and optimizer and loss function results in changing the accuracy of the model. Similarly, if we change the batch size and number of epochs, then also the precision and scores will change. So, we can even try to improve this accuracy by finding the best parameters for the model. 
→ We used high-level GPUs provided by Nvidia for running our codes, which helped in speeding up the training process.

**Conclusion**

The CNN classification is really useful for classifying the given time series data. The main step of this Model is to convert the time series data to frequency domain data using CQT resulting in the formation of spectrogram images. Here, the main reason for this conversion is, we need to identify if there is a GW signal present along with noise in the signal measured by the observatory. Because the time-series data shows us only how the signal is transforming along with time, it does not help us to classify the signals, whereas in the case of the frequency domain, the spectrogram formed shows us how much of each signal is present in the given data. This means if a GW signal is present, then the spectrogram image also changes along with it, so that we can identify if there is a gravitational wave present or not. We have used basic concepts of CNN layers and Tensorflow to get this model.
Also, data preprocessing is a crucial step. One of the biggest challenges in this problem is to handle such large datasets, we had overcome this problem using Tensorflow operations. The model we got is performing well just after 3 epochs with an accuracy of 0.83. Here, we stacked the 3 signals side by side and formed a single spectrogram image for making the training easy. It would be interesting to see if we consider each signal from each observatory as a separate feature and implement this model. Also, we learned to use GPU’s which helped us to train the models quickly. Just with a help of a simple CNN model we are able to achieve good accuracy, if we had a better understanding of the working of neural networks at deeper levels, we can even try to increase this accuracy. 



