#  Harry Polishook hp482@njit.edu
#  CS643 Summer 2020 
#  Project 2

Github: https://github.com/hp482/cs643-project2

Docker Container: https://hub.docker.com/repository/docker/hp482/cs643

Training Environment: 

Amazon ECM, 4 worker nodes m5.xlarge,  with Jupyter.    
 
![Image of Cluster](https://github.com/hp482/cs643-project2/blob/master/cluster.jpg)


Prediction Environment: 

# To run in docker 
Docker pull hp482/cs643:1  

Windows: docker run --rm  -v "%cd%":/data  hp482/cs643:1  <filename>

Linux: docker run --rm -v $(pwd):/data hp482/cs643:1 <filename>

#To run stand-alone:

Application requires python, pyspark, spark 2.4.6, Hadoop 2.7.3, AWS Java SDK 1.7.4, numpy  and supporting installations.   

cs643.py  <Dataset.csv>
