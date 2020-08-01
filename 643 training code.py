# ########################################################################################################
#  643 trainng code extract.py 
#
#  Harry Polishook hp482@njit.edu
#  CS643 Summer 2020 
#  Project 2, Model Training Code Extract
#
# ########################################################################################################

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
dataset = spark.read.option("delimiter", ";").csv('TrainingDataset.csv',inferSchema=True, header =True)
validationdataset = spark.read.option("delimiter", ";").csv('ValidationDataset.csv',inferSchema=True, header =True)

# ################################################################################################################

dataset.printSchema()
validationdataset.printSchema()
print (dataset.columns[1])




# ################################################################################################################
# Process Input Files 
# because of the data file format use the filename list rather than field names explicitly 
# for reference here's the expected column names
# TrainingDataset.csv': b'"""""fixed acidity"""";""""volatile acidity"""";""""citric acid"""";""""residual sugar"""";""""chlorides"""";
# """"free sulfur dioxide"""";""""total sulfur dioxide"""";""""density"""";""""pH"""";""""sulphates"""";""""alcohol""""

	
# Input all the features in one vector column
assembler = VectorAssembler(inputCols=[dataset.columns[1], dataset.columns[2], dataset.columns[3], dataset.columns[4], dataset.columns[5], dataset.columns[6], dataset.columns[7], dataset.columns[8], dataset.columns[9],dataset.columns[10] ], outputCol = 'Attributes')
output = assembler.transform(dataset)


finalized_data = output.select("Attributes",dataset.columns[11] )
finalized_data.show()

valid_output = assembler.transform(validationdataset)

valid_finalized_data = valid_output.select("Attributes",validationdataset.columns[11] )
valid_finalized_data.show()

# 80/20 split train / test 
train_data,test_data = finalized_data.randomSplit([0.8,0.2])
regressor = LinearRegression(featuresCol = 'Attributes', labelCol = dataset.columns[11] )

#Train mdoel with training split 
regressor = regressor.fit(train_data)

pred = regressor.evaluate(test_data)

#Predict the model
pred.predictions.show()

predictions = regressor.transform(valid_finalized_data)
predictions.show()


dataset.groupby("quality").count().show()

# ################################################################################################################
# export the trained model and create a zip file for ease of download
import shutil
from pyspark.ml.regression import LinearRegressionModel
regressor.write().overwrite().save("cs643")

path_drv = shutil.make_archive("cs643", format='zip', base_dir="cs643")
shutil.unpack_archive("cs643.zip", "test",format='zip',)

loadedRegressor = LinearRegressionModel.load("test/cs643")
predictions = loadedRegressor.transform(valid_finalized_data)
print(loadedRegressor.numFeatures)
predictions.show()


# ################################################################################################################
# run some equick evaluations 
from pyspark.ml.evaluation import RegressionEvaluator
eval = RegressionEvaluator(labelCol= dataset.columns[11], predictionCol="prediction", metricName="rmse")
# Root Mean Square Error
rmse = eval.evaluate(pred.predictions)
print("RMSE: %.3f" % rmse)
# Mean Square Error
mse = eval.evaluate(pred.predictions, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)
# Mean Absolute Error
mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)
# r2 - coefficient of determination
r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
print("r2: %.3f" %r2)
