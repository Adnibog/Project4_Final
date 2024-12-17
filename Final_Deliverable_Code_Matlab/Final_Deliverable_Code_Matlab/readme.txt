Readme for Project 4:

Part 1:
The DimensionalityReduction.m file is set to run the PCA analysis for the images.  You will need to open the function in Matlab and hit run.  The script will ask for the folder location for the images and then search for the PGM images.  Within this script, you can change the PCA type from 1=Eigen to 2=SVD, just like we did in class.  The script will then run and give a mean image, the variance graph, and then the top eigenfaces.  


Part 2:
The KNN.m and SVM.m files use the DimensionalityReduction.m as a backbone and function the same way with regards to requiring the user input for the image folder.  Once the folder has ben selected, the image classification will run for 35 classes training with 8 images and testing with the final two.  The output will be the face results and you will see an accuracy in the Matlab command window.  