function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);
height = layer.num;
width = input.width;
channel = input.channel;
batch_size = input.batch_size;

% Replace the following line with your implementation.
output.data = zeros([n, k]);
output.height = height;
output.width = width;
output.channel = channel;
output.batch_size = input.batch_size;

w = param.w';
x = input.data;
b = param.b';

for i = 1:k
    
    output.data(:,i) = w * x(:,i) + b;
    
end
