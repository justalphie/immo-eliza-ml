# Model card
## Project context
This is a study project aimed at training a machine learning model with the usage of ```scikit-learn``` library. The goal of the project was to train the model on the dataset that has been collected within the framework of [immo-eliza-scraping](https://github.com/justalphie/immo-eliza-scraping-immozila) project and analysed during the [Cleaning / EDA](https://github.com/justalphie/immo-eliza-scraping-immozila-Cleaning-EDA) project.
## Data
The input dataset of the model consists of two csv files, one containing observations on apartments, and the other one for houses. The number of observations is around 9000 for each. The data include property id, locality name, postal code, latitude, longitude, property type, property subtype, price, type of sale, number of rooms, living area, kitchen type, fully equipped kitchen, furnished, open fire, terrace, terrace area, garden, garden area, surface of good, number of facades, swimming pool, state of building, main city, province. 
## Model details
A number of models have been tested for the project. These include: 

- LinearRegression
- Ridge
- DecisionTreeRegressor
- StackingRegressor
- GradientBoostingRegressor
- RandomForestRegressor

It was decided that in predict.py the Ridge model with modified preprocessing steps will be implemented. It combines short training time with good quality of predictions.

## Performance
Numerous models have been tested in the project. 
### The models trained on house data

| Model                                                                  | Preprocessing version | Train score | Test score |
|------------------------------------------------------------------------|-----------------------|-------------|------------|
| StackingRegressor(GradientBoostingRegressor and RandomForestRegressor) | 1                     | 0.79        | 0.73       |
| StackingRegressor(Ridge and GradientBoostingRegressor)                 | 2                     | 0.8         | 0.73       |
| Ridge                                                                  | 2                     | 0.73        | 0.72       |
| GradientBoostingRegressor                                              | 2                     | 0.79        | 0.72       |
| GradientBoostingRegressor                                              | 1                     | 0.78        | 0.71       |
| StackingRegressor(Ridge and DesicionTreeRegressor)                     | 1                     | 0.67        | 0.67       |
| Ridge                                                                  | 1                     | 0.61        | 0.66       |
| Lasso                                                                  | 1                     | 0.61        | 0.66       |
| RandomForestRegressor(min_samples_leaf=15)                             | 1                     | 0.65        | 0.59       |
| ElasticNet                                                             | 1                     | 0.4         | 0.41       |
| DecisionTreeRegressor                                                  | 1                     | 0.99        | 0.41       |
| LinearRegression                                                       | 1                     | 0.52        | -1.33      |

### The models trained on apartment data
| Model                                                                  | Preprocessing version | Train score | Test score |
|------------------------------------------------------------------------|-----------------------|-------------|------------|
| StackingRegressor(GradientBoostingRegressor and RandomForestRegressor) | 1                     | 0.79        | 0.73       |
| StackingRegressor(Ridge and GradientBoostingRegressor)                 | 2                     | 0.8         | 0.73       |
| Ridge                                                                  | 2                     | 0.73        | 0.72       |
| GradientBoostingRegressor                                              | 2                     | 0.79        | 0.72       |
| GradientBoostingRegressor                                              | 1                     | 0.78        | 0.71       |
| StackingRegressor(Ridge and DesicionTreeRegressor)                     | 1                     | 0.67        | 0.67       |
| Ridge                                                                  | 1                     | 0.61        | 0.66       |
| Lasso                                                                  | 1                     | 0.61        | 0.66       |
| RandomForestRegressor(min_samples_leaf=15)                             | 1                     | 0.65        | 0.59       |
| ElasticNet                                                             | 1                     | 0.4         | 0.41       |
| DecisionTreeRegressor                                                  | 1                     | 0.99        | 0.41       |
| LinearRegression                                                       | 1                     | 0.52        | -1.33      |


## Limitations
The R-squared score of the model is limited to 0.73. This only allows for very rough estimations of the price. 

Possible improvements are likely to take place if more features were considered, e.g. the energy score of the property, or imformation about the surroundings (schools, public transport, etc.).

The model was only trained on the 80% of the dataset of each type (houses and apartments). Increased sample is highly likely to improve the performance.

## Usage
To use the model, please check the ```requirements.txt``` file. Necessary libraries include Scikit-learn, pandas, numpy, pickle.

To make price predictions, use ```predict.py```. Please specify the folder: ```houses``` or ```apartments```, depending on what type of properties you would like to evaluate. Please check the format of the input dataset to carry out the prediction. The predictions of the prices will be written to the ```y_predict.csv``` in the folder specified earlier.

To train the model, use ```train.py```.  Type ```houses``` or ```apartments``` depending on your dataset. The model's ```.pickle``` file  will be saved in the ```models``` subfolder, along with the ```txt``` file containing the train and the test scores of the model.

## Maintainers
Should you have any questions, please contact [Alfiya Khabibullina](https://github.com/justalphie)