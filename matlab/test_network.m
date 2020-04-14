
%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat
confusion = zeros(10,10);
%% Testing the network
% Modify the code to get the confusion matrix

for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [val,index] = max(P);
    prediction(i:i+99) = index;
end
for i=1:size(xtest, 2)
    label = ytest(i);
    pre = prediction(i);
    confusion(label,pre) = confusion(label,pre) + 1;
end
disp("Confusion Matrix")
disp(confusion)