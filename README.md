# IndianBirdsClassification

Here are two important Notebook

1. [IndianBirdsClassifier.ipynb](https://github.com/aman5319/BirdsClassification/blob/master/IndianBirdsClassifier.ipynb)  
This notebook consist of Image classifier built using Fast AI library, steps performed in this notebook

	​		1. [Creating your own dataset from Google Images](https://render.githubusercontent.com/view/ipynb?commit=58c23e3dc66ca42896b1a23e776be3d59fdbd3a6&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4149362d42616e67616c6f72652d436861707465722f323031382d6379636c652d322f353863323365336463363663613432383936623161323365373736626533643539666462643361362f53657373696f6e732f53657373696f6e5f31322f7068617365315f73616d706c652e6970796e62&nwo=AI6-Bangalore-Chapter%2F2018-cycle-2&path=Sessions%2FSession_12%2Fphase1_sample.ipynb&repository_id=143403708&repository_type=Repository#Creating-your-own-dataset-from-Google-Images)   

	​		2. Use transfer learning on ResNet34 model trained on Imagenet

	​		3. Use Lr find and One cycle policy method to get faster results .

	​		Finally The model accuracy is **91%**  

------
2. [Creating_Datasets.ipynb](https://github.com/aman5319/BirdsClassification/blob/master/Creating_Datasets.ipynb)  

   ​	All the above steps defined in the above IndianBirdsClassifier.ipynb are done using Fastai library, where all we do is just simple library method calls and we get a very good result

   ​	But What this repo focuses on replicating all of those into Keras

   ​	So This Notebook shows  how to easily create an image dataset through Google Images and load them in to keras to train your model .
_____

Birds.zip file contains 10 CSV files of different birds. With each CSV file having the image URL of the birds.

3. [LrFinder.py](https://github.com/aman5319/BirdsClassification/blob/master/LrFinder.py) 

   ​	This class uses the Cyclic Learning Rate history to find a set of learning rates that can be good  initializations for the One-Cycle training proposed by Leslie Smith in the paper referenced  below. 
   
   ​	A port of the Fast.ai implementation for Keras.
   
   ​	Interpretation
   
   ​	Upon visualizing the loss plot, check where the loss starts to increase rapidly. Choose a learning rate at somewhat prior to the corresponding position in the plot for faster convergence. This will be the max_lr.
   
   ​        References:
   ​            [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)

   ```python
   #usage
   from LrFinder import LRFinder
   
   lrfind = LRFinder(max_iteration = len(feature_train)//batch_size )
   history = classifier.fit(feature_train,
                            label_train,
                            epochs=1,
                            batch_size=batch_size,
                            shuffle=True ,
                            callbacks=[lrfind] )
   
   ```

4. [OneCyclePolicy.py](https://github.com/aman5319/BirdsClassification/blob/master/OneCyclePolicy.py)

   ​	This callback implements a cyclical learning rate policy (CLR). This is a special case of Cyclic Learning Rates, where we have only 1 cycle. After the completion of 1 cycle, the learning rate will decrease rapidly to 10000th its initial lowest value.
      Reference

   ​	[A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
   ​	[Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)

   ```python
   #usage
   from OneCyclePolicy import OneCycleScheduler
   
   fit_one_cycle = OneCycleScheduler(num_iteration = len(feature_train)//batch_size ,
                                     num_epochs =4 ,
                                     max_lr = 2e-3) #max_lr is the lr you got from  lrfind
   classifier.fit(feature_train,
                  label_train ,
                  epochs=4,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=[fit_one_cycle],
                  validation_data=(feature_test,label_test))
   
   ```


The [Mnist_callbacks_test.ipynb](https://github.com/aman5319/BirdsClassification/blob/master/Mnist_callbacks_test.ipynb) file contains the test of above two callbacks
