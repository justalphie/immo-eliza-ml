# The immo-eliza-ml machine learning project

## Project description

<img align="right" height="200" src="https://assets.everspringpartners.com/dims4/default/5c1df5b/2147483647/strip/true/crop/1400x800+0+0/resize/1600x914!/format/webp/quality/90/?url=http%3A%2F%2Feverspring-brightspot.s3.us-east-1.amazonaws.com%2Ffe%2F06%2Ff23661be455e97d009c6ae418995%2Freal-estate-finance.jpg" />

Real estate business needs fast and efficient tools to take advantageous decisions. Automatic price estimator is a handy tool that can boost the work productivity of real estate projects and navigate the clients in the sea of real estate offers. 

The immo-eliza-ml program is an example of such a tool. With the help of such features as location and living area of the apartment of the house it can rapidly predict the price of the property. 

## Usage
To use the model, please check the ```requirements.txt``` file. Necessary libraries include **Scikit-learn, pandas, numpy, pickle**.

To make price predictions, use ```predict.py```. Please specify the folder: ```houses``` or ```apartments```, depending on what type of properties you would like to evaluate. Please check the format of the input dataset to carry out the prediction. The predictions of the prices will be written to the ```y_predict.csv``` in the folder specified earlier.

To train the model, use ```train.py```.  Type ```houses``` or ```apartments``` depending on your dataset. The model's ```.pickle``` file  will be saved in the ```models``` subfolder, along with the ```txt``` file containing the train and the test scores of the model.


## Structure
Example structure of the folder with input and output files 

```
├───houses
│   ├───data
│   │       dataset.csv
│   │       X_test.csv
│   │       X_train.csv
│   │       y_predict.csv
│   │       y_test.csv
│   │       y_train.csv
│   │
│   ├───models
│   │       ridge.pickle
│   │
│   └───preprocessings
│           preprocessings.pickle
```

## Project timeline
The project was carried out within the framework of Data&AI training by [BeCode](https://becode.org/) within 6 days.
- Day 1-2 data preprocessing
- Day 3 model training
- Day 4 model evaluation 
- Day 5 structuring
- Day 6 project finalization

## Authors
The program was developed by [Alfiya Khabibullina](https://github.com/justalphie) under the supervision of the coach [Vanessa Rivera-Quinones](https://github.com/vriveraq)
