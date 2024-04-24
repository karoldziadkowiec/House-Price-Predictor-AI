# House Price Predictor
![0](github-img/0.jpg)

## About project
The project is a university assignment implemented in a **Jupyter Notebook** environment. It aims to present a house price predictor specifically tailored for the Polish real estate market based on **neural networks**. The model tries to forecast house prices with high accuracy.

## Dataset
Kaggle page: https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland

## Technologies
- Tools:
     - **Google Colab** with **Jupyter Notebook**
- Programming language: 
     - **Python**
- Libraries:
     - **Pandas** - for data manipulation and analysis
     - **NumPy** - for numerical calculations
     - **Matplotlib** - for creating charts and data visualization
     - **Seaborn** - for more advanced data visualizations
     - **Plotly Express** - for interactive charts
     - **TensorFlow** - for building and training neural networks
     - **Scikit-learn** - for data partitioning, standardization, model evaluation and outlier removal
- Framework:
     - **Keras** - for building neural network models
- Technology:
     - **Neural networks** - for predicting real estate prices
- Algorithm:
     - **Regression**
- Dataset source:
     - **Kaggle** (**csv** file)

## Solution
Imported data:

![1](github-img/1.png)

Displaying loaded properties on the map:

![2](github-img/2.png)

Price per square meter histogram:

![3](github-img/3.png)

Boxplot chart of prices per square meter in individual cities:

![4](github-img/4.png)

### Division of data into input and output variable(s):

Input variables:
- Quantitative variables: **floor, latitude, longitude, rooms, sq, year, city_Krakow, city_Poznan, city_Warszawa**
- Qualitative variables: **-**

Output variable:
- **price** (quantitative variable)

```bash
X = data[['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year', 'city_Krakow', 'city_Poznan', 'city_Warszawa']]
y = data['price']
```

Histogram for the output variable (price):

![5](github-img/5.png)

### Problem solved

The problem being solved in this case is **regression**. The goal is to predict property prices based on various characteristics such as location, number of rooms, square footage, year of construction, etc. In regression, we try to find the relationship between the independent variables (features) and the dependent variable (price) in order to predict the numerical value, i.e. the price of the property .

### Dividing data into sets
- **Training**: used to train the model.
- **Test**: used to evaluate the performance of the model after it has been trained.

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Neural networks used
- *Network type:* **Feedforward** neural network (unidirectional), used in regression.
- *Network Architecture:* **5 different neural network architectures**, each consisting of dense layers. Each model differs in the number of layers and the number of neurons in each layer.
- *Number of training epochs:* Each model is trained for **100 epochs**.
- *Network training method:* First, a set of models is created with different neural network architectures. The network is trained using the **Adam** optimizer, which adjusts the network's weighting factors to minimize the mean squared error (**MSE**) function. The error function used for training is **MSE** (Mean Squared Error), and **MAE** (Mean Absolute Error) and **R2 Score** are also used as evaluation metrics. Each model is trained on the training data for **100 epochs** using a batch size of **32**. Additionally, **20%** of the training data is used as the **validation set**. After training, each model is evaluated on the training and test sets. The prediction results for the test set are visualized using a scatter plot, where the axes represent the actual values and predictions. The ideal model should appear as a straight line that perfectly aligns with the points.

```bash
models = []
```

**Model 1**:

A **Sequential** model, which means that the layers are arranged one after the other in a sequence. It consists of **3 dense layers**. The first and second layers each have **64 neurons and ReLU activation**, which means that it applies a non-linear function **ReLU** (Rectified Linear Unit) on its outputs. The third layer has **1 neuron**, which suggests that the model predicts one output value.

```bash
model1 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
models.append(model1)
```

**Model 2**:

**Sequential** type model, consisting of **4 dense layers**. The first and second layers each have **128 neurons and ReLU activation** (uses the non-linear ReLU function on its outputs). The third layer has **64 neurons with ReLU activation**. The fourth layer has **1 neuron**.

```bash
model2 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
models.append(model2)
```

**Model 3**:

**Sequential** type model, consisting of **3 dense layers**. The first and second layers each have **64 neurons and ReLU activation** (applies a non-linear ReLU function to its outputs). The third layer is a **dropout layer with a value of 0.2**, which means it randomly disables **20%** of neurons during training to prevent overfitting**. The last layer has 1 neuron.

```bash
model3 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
models.append(model3)
```

**Model 4**:

**Sequential** type model, consisting of **4 dense layers**. The first layer has **64 neurons with ReLU activation**, the second layer has **128 neurons also with ReLU activation**, the third layer has **64 neurons with ReLU activation**, and the fourth layer has **1 neuron** without any activation.

```bash
model4 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
models.append(model4)
```

**Model 5**:

**Sequential** type model, consisting of **5 dense layers**. The first four layers have **64 neurons with ReLU activation**. The fifth layer has **1 neuron** with no activation at all.

```bash
model5 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
models.append(model5)
```


### Training models and displaying results

```bash
for i, model in enumerate(models, start=1):
    print(f'Model {i}')

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    loss_train, mae_train = model.evaluate(X_train, y_train, verbose=0)
    loss_test, mae_test = model.evaluate(X_test, y_test, verbose=0)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\nZBIÓR TRENINGOWY(UCZĄCY):")
    print('Średnia wartość bezwzględna róznicy(MAE)', mae_train)
    print('Średnia wartość kwadratu różnicy(MSE):', loss_train)
    print('Średni błąd modelu(RMSE):', np.sqrt(loss_train))
    print("Współczynnik determinacji(R2):", r2_score(y_train, y_pred_train))

    print("\nZBIÓR TESTUJĄCY:")
    print('Średnia wartość bezwzględna róznicy(MAE)', mae_test)
    print('Średnia wartość kwadratu różnicy(MSE):', loss_test)
    print('Średni błąd modelu(RMSE)::', np.sqrt(loss_test))
    print("Współczynnik determinacji(R2):", r2_score(y_test, y_pred_test))

    plt.figure(figsize=(16, 8))
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(0, 7000000)
    plt.ylim(0, 7000000)
    _ = plt.plot([0, 7000000], [0, 7000000])
    plt.show()
```

**Model 1**:
- **Training set:**
    * MAE: 133382.21875
    * MSE: 67452256256.0
    * RMSE: 259715.7220038864
    * R2: 0.6799250734982716
- **Test set:**
    * MAE: 131144.328125
    * MSE: 65625067520.0
    * RMSE: 256173.9009345019
    * R2: 0.7059435631757701

![6](github-img/6.png)


**Model 2**:
- **Training set:**
    * MAE: 127108.5625
    * MSE: 60705611776.0
    * RMSE: 246385.0883799586
    * R2: 0.7119394830095441
- **Test set:**
    * MAE: 125159.0859375
    * MSE: 58691358720.0
    * RMSE: 242262.99494557563
    * R2: 0.7370125469520319

![7](github-img/7.png)


**Model 3**:
- **Training set:**
    * MAE: 132783.75
    * MSE: 67626471424.0
    * RMSE: 260050.90160197485
    * R2: 0.6790983662673938
- **Test set:**
    * MAE: 130958.796875
    * MSE: 65878700032.0
    * RMSE: 256668.46325951305
    * R2: 0.7048071599283495

![8](github-img/8.png)


**Model 4**:
- **Training set:**
    * MAE: 126462.9453125
    * MSE: 61010079744.0
    * RMSE: 247002.18570692852
    * R2: 0.7104945113348253
- **Test set:**
    * MAE: 124352.7578125
    * MSE: 59110133760.0
    * RMSE: 243125.7570887955
    * R2: 0.7351361058748898

![9](github-img/9.png)


**Model 5**:
- **Training set:**
    * MAE: 123051.4453125
    * MSE: 56869249024.0
    * RMSE: 238472.74272754948
    * R2: 0.7301435949288448
- **Test set:**
    * MAE: 121221.3515625
    * MSE: 53265031168.0
    * RMSE: 230792.18177399336
    * R2: 0.7613271683648691

![10](github-img/10.png)


### Conclusions

**Model 5** appears to be the best model, achieving the lowest errors and the highest coefficient of determination on both datasets. The coefficient of determination (*R2*) for all models is around **0.70**, which means that the models are able to explain about **70%** of the data variability, which is moderately satisfactory. Models **2, 4 and 5** show better results than models **1 and 3** on both the training and test sets, expressed in lower errors (*MAE, MSE, RMSE*) and higher coefficient of determination (*R2*). Models **2, 4 and 5** may be preferred models. It is worth noting that although there is some improvement in subsequent models, the differences in results are not significant, suggesting that further model testing may be necessary to obtain more significant improvements. In summary, the results are not perfect, but they are satisfactory.


### Proposals for further development of the project
- Further testing of different neural network architectures and hyperparameters to find the optimal model with better results.
- Testing other machine learning algorithms that can perform well on housing price data and predictions.
- Exploration of additional features that can improve the quality of predictions.
- Applying more advanced optimization techniques such as network density optimization to optimize model performance.
- Use of advanced regularization techniques to avoid overtraining.