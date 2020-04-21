function theta=trainingNeuralNetwork(initial_theta_value,X,y,lambda,input_layer_size,hidden_layer_size,num_labels,maxiter)

options=optimset('GradObj','on','MaxIter',maxiter,'Display','iter');
initial=initial_theta_value;

for i=0:100:60000
    X=X(i+1:i+100,:);
    y=y(i+1:i+100,:);
[theta,fval,exitflag]=fminunc(@(t)(nnCostFunction(t,input_layer_size,hidden_layer_size,num_labels,X(1:10,:),y(1:10,:),lambda)),initial,options);
initial=theta;
end
end