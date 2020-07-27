# ########################################################################################################
#  643.py 
#
#  Harry Polishook hp482@njit.edu
#  CS643 Summer 2020 
#  Project 2, prediction part for Docker 
#
# ########################################################################################################



import sys
import shutil
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import os.path
from os import path



# ########################################################################################################
def usageMessage(): 
	print("Usage:  643.py <filename>,  remember to mount the local directory if running in docker.")
	print("Usage:  and datafile must be in current directory.  ")
	print("Usage:  Windows: docker run --rm -v ""%cd%"":/data hp482/cs643:1 <filename> ")
	print("Usage:  PowerShell: docker run --rm  -v ${PWD}:/data hp482/cs643:1 <filename>")
	print("Usage:  Linux: docker run --rm -v $(pwd):/data hp482/cs643:1 <filename>")
	
	
# ########################################################################################################
def main():
		

	
		
		
	if len(sys.argv) > 1:
		predictionFile = sys.argv[1]
		if path.isfile(predictionFile):
			
			print("Processing File "+predictionFile)
		else:
			print("File not found "+ predictionFile)
			usageMessage()
			exit()
	else:
		usageMessage()
		

	spark = SparkSession.builder.master("local[*]").getOrCreate()


	# load trained model 
	loadedRegressor = LinearRegressionModel.load("/cs643")
	# read dataset to predict 
	validationdataset = spark.read.option("delimiter", ";").csv(predictionFile,inferSchema=True, header =True)

	# validationdataset.printSchema()


	# Process the data set into expected format 
	
	# combine the first 10 columns into attributes.
	# because of the data file format use the filename list rather than field names explicitly 
	# for reference here's the expected column names
	# TrainingDataset.csv': b'"""""fixed acidity"""";""""volatile acidity"""";""""citric acid"""";""""residual sugar"""";""""chlorides"""";
	# """"free sulfur dioxide"""";""""total sulfur dioxide"""";""""density"""";""""pH"""";""""sulphates"""";""""alcohol""""
	
	assembler = VectorAssembler(inputCols=[validationdataset.columns[1], validationdataset.columns[2], validationdataset.columns[3], validationdataset.columns[4], validationdataset.columns[5], validationdataset.columns[6], validationdataset.columns[7], validationdataset.columns[8], validationdataset.columns[9],validationdataset.columns[10] ], outputCol = "Attributes")

	valid_output = assembler.transform(validationdataset)

	valid_finalized_data = valid_output.select("Attributes",validationdataset.columns[11])
	# valid_finalized_data.show()
	
	# predict the quality
	predictions = loadedRegressor.transform(valid_finalized_data)
	
	# display results
	predictions.show(2000,  truncate = False)  # we could do this on a row count rather than 2000, but what if we end up with million row model somehow


if __name__== "__main__":
   main()
