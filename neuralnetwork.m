function neuralnetwork(hidden_layer_size,num_train_images,num_test_images,MaxIter,reg_parameter)

if~exist('hidden_layer_size','var')||isempty(hidden_layer_size)
    hidden_layer_size=25;
end

if~exist('num_train_images','var')||isempty(num_train_images)
    num_train_images=60000;
end

if~exist('num_test_images','var')||isempty(num_test_images)
    num_test_images=10000;
end
if ~exist('MaxIter', 'var') || isempty(MaxIter)
    MaxIter = 30;
end

if ~exist('reg_parameter', 'var') || isempty(reg_parameter)
    reg_parameter = 0.1;
end

input_layer_size=784;
num_labels=10;

load('MNISTDataset');
X=trainingImages((1:num_train_images),:);
y=trainingLabels((1:num_train_images),:);
testX=testImages((1:num_test_images),:);
testy=testLabels((1:num_test_images),:);

m=size(X,1);
n=size(testX,1);

sel=randperm(size(X,1));
sel=sel(1:100);

%displayData(X(sel,:));
fprintf('Please hit enter to continue');
pause;



Theta1=randInitializeWeights(input_layer_size,hidden_layer_size);
size(Theta1);
Theta2=randInitializeWeights(hidden_layer_size,num_labels);
size(Theta2);

nn_params=[Theta1(:);Theta2(:)];

J=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,0);
J1=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,3);


checkNNGradients;
checkNNGradients(1);

validation_ratio=input('enter the ratio for validation: ');
[lambda_vec,error_train,error_val]=validationCurve(X, y,nn_params,input_layer_size,hidden_layer_size,num_labels,MaxIter,validation_ratio);

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

combined_matrix=[lambda_vec,error_train,error_val];
best_lambda=sortrows(combined_matrix,[3,2]);
reg_parameter=best_lambda(1,1);

Theta=trainingNeuralNetwork(nn_params,X,y,reg_parameter,input_layer_size,hidden_layer_size,num_labels,200);


Theta1 = reshape(Theta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(Theta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 
             
save('saveweights.mat','Theta1','Theta2'); 

prediction_training=predict(Theta1,Theta2,X);
prediction=predict(Theta1, Theta2, testX);

test_accuracy=mean(double(prediction_training));
training_accuracy=mean(double(prediction));

fprintf("Test set data Accuracy:%f",test_accuracy);
fprintf("Training Accuracy: %f",training_accuracy);



end







