% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray


%data_path = 'LabHOG/data/'; %change if you want to work with a network copy
%non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
%feature_params = struct('template_size', 36, 'hog_cell_size', 6);

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

template_size = feature_params.template_size ;
hog_cell_size = feature_params.hog_cell_size ;

num_cells = (template_size/hog_cell_size);
dimensionality = (num_cells^2) * 31 ;

num_samples_per_image = ceil(num_samples/num_images) ;
num_samples_new = num_samples_per_image*num_images ;
features_neg = zeros(num_samples_new,dimensionality) ;


for i=1:num_images
    %disp(i)
    fprintf('Detecting random negative features in %s\n', image_files(i).name)
    image_name = image_files(i).name ;
    image_path = fullfile(non_face_scn_path,image_name) ;
    img = imread(image_path) ;
    
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    [img_heigth, img_width] = size(img)
    %img = imresize(img,[36 36]) ;
    for j=1:num_samples_per_image
        rand_x_position = ceil(rand*(img_width-template_size-1)) ;
        rand_y_position = ceil(rand*(img_heigth-template_size-1)) ;
        image_cropped = img(rand_y_position:rand_y_position+template_size,rand_x_position:rand_x_position+template_size) ;
        hog = vl_hog(im2single(image_cropped),hog_cell_size);
        
        hog_reshaped = reshape(hog,[1,dimensionality]) ;
        features_neg(((i-1)*num_samples_per_image)+j,:) = hog_reshaped ;
    end
end