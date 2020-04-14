new_image = imread('../images/Images/all2.png');

layers = get_lenet();
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat
layers{1}.batch_size = 1;
if size(new_image,3) > 1
    new_image = rgb2gray(new_image);
end
img = new_image;

imshow(new_image)

imgs = [];
max_y = 0;
min_y = 1000000;
min_i = 1000000;
draw = 0;

max_w = [];
for i = 1:size(new_image,2)
    count = 0;
    for j = 1:size(new_image,1)        
        if new_image(j,i) > 100
            count = count + 1;            
            if j < min_y                
                min_y = j;                
            end            
            if max_y < j                 
                max_y = j;                
            end
        end
    end
    
    if and(count == 0, draw)
        im = new_image(min_y:min_y+35,:);
        minx = i;
        max_w = [max_w,min_i,i-min_i];
        max_y = 1;
        min_y = 1000000;
        min_i = 1000000;
        
        draw = 0;
    end
    if count ~= 0
        if i < min_i
            
            min_i = i;
        end
        draw = 1;
    end
end

max_y = 0;
min_y = 1000000;
min_i = 1000000;
draw = 0;

max_h = [];
for i = 1:size(new_image,1)
    count = 0;
    for j = 1:size(new_image,2)        
        if new_image(i,j) > 100
            count = count + 1;            
            if j < min_y                
                min_y = j;                
            end            
            if max_y < j                 
                max_y = j;                
            end
        end
    end
    
    if and(count == 0, draw)
        im = new_image(min_y:min_y+35,:);
        minx = i;
        max_h = [max_h,min_i,i-min_i];
        max_y = 1;
        min_y = 1000000;
        min_i = 1000000;
        
        draw = 0;
    end
    if count ~= 0
        if i < min_i
            
            min_i = i;
        end
        draw = 1;
    end
end

count = 1;
for i=1:2:size(max_h,2)
    
    for j = 1:2:size(max_w,2)

        
        rectangle('Position', [max_w(j),max_h(i),max_w(j+1),max_h(i+1)], 'EdgeColor','w');
        
        th = 20;

        im = new_image(max_h(i)-th:max_h(i) + max_h(i+1)+th, max_w(j)-th:max_w(j)+max_w(j+1) +th);
        im = imresize(im,[28,28]);
        imgs{count} = im;
        count = count + 1;
    
    end
    
end

for x = 1:length(imgs)

    im = imgs{x};
    im = imresize(im,[28,28]);
    for i = 1:size(im,1)

        for j = 1:size(im,2)

            if im(i,j) > 100
                im(i,j) = 255;
            else
                im(i,j) = 0;

            end

        end
    end

    [output, P] = convnet_forward(params, layers, reshape(im',[28*28,1]));
    [val,index] = max(P);
    disp("The prediction of the image is :-  ")
    disp(index-1)
    
end