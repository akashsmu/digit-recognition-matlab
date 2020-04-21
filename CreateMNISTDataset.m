clear;close all;clc
trainingImages=loadMNISTImages('train-images.idx3-ubyte');
trainingLabels=loadMNISTLabels('train-labels.idx1-ubyte');

trainingLabels(trainingLabels==0)=10;

testImages=loadMNISTImages('t10k-images.idx3-ubyte');
testLabels=loadMNISTLabels('t10k-labels.idx1-ubyte');

testLabels(testLabels==0)=10;

save('MNISTDataset');
