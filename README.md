# classification problem
 use ANN to classify the music data

 Task1: load the data and label from data_task1.npy and label_task1.npy. X is the data, y is the target. Then create the class for ANN, use nn.Linear to apply a linear transformation to the data with 20 input and 16 output to the next layer and so on, the number of final output will be 4. Then define a function called forward to apply the element-wise function and store it in x for further usage. After that, use the loss function, CrossEntropyLoss for training the classification problem. 

 Task2:load the data and label from data_task2.npy and label_task2.npy. X is the data, y is the target. Then, create the class for ANN, use nn.Linear to apply a linear transformation to the data with 100 input features and 16 output to the next layer and so on, the number of final output will be 16. Then define a function called forward to apply the element-wise function and store it in x for further usage. Also, in the “forward” function, flatten a contiguous range of dims in a tensor. After that, usee the loss function, CrossEntropyLoss for training the classification problem.
