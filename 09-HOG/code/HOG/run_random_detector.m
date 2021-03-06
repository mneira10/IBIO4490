% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

%data_path = 'LabHOG/data/'; %change if you want to work with a network copy
%test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
%feature_params = struct('template_size', 36, 'hog_cell_size', 6);

%b_struct = load('b_pos_6713_neg_10138_lamba_0_0001.b')
%b = b_struct.b

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

%%%%% parameters
template_size = feature_params.template_size;
hog_cell_size = feature_params.hog_cell_size;
num_cells = template_size / hog_cell_size;
dimensionality = (num_cells^2) * 31 ;

%scales = 2.0 ;
scales = [1,0.9,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.2,0.1] ;
%num_scales = length(scales) ;
threshold = -1.0 ;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);



for i = randi([1 length(test_scenes)],1)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = imresize(img,[256,256]) ;
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    %initialize these as empty and incrementally expand them.
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);

    for scale = scales
        img_scaled = imresize(img, scale);
        %size(img_scaled)
        [img_scaled_height, img_scaled_width] = size(img_scaled) ;
        
        hog = vl_hog(img_scaled, hog_cell_size);
        %size(hog)
        num_cell_hog_y = size(hog, 2) ;
        num_cell_hog_x = size(hog, 1) ;
        
        for i_x = 1:num_cell_hog_x - num_cells - 1
            for i_y = 1:num_cell_hog_y - num_cells - 1
                %i_x
                %i_y
                hog_temp = hog(i_x:i_x+num_cells-1, i_y:i_y+num_cells-1, :);
                %size(hog_temp)
                hog_temp_reshaped = reshape(hog_temp,[1,dimensionality]) ;
                temp_confidence = sum(hog_temp_reshaped.*w') + b ;
                
                if temp_confidence > threshold
                    %temp_bboxes = [i_x, i_y, i_x+num_cells-1, i_y+num_cells-1];
                    %disp(temp_bboxes)
                    %temp_bboxes = temp_bboxes * hog_cell_size * (1/scale);
                    %disp(temp_bboxes)
                    %disp(scale)
                    %imshow(img_scaled)
                    %prompt = 'What is the original value? ';
                    %x = input(prompt)
                    
                    temp_boxes = [i_y, i_x, i_y+num_cells-1, i_x+num_cells-1];
                    temp_boxes = temp_boxes * hog_cell_size * (1/scale);
                    cur_bboxes = [cur_bboxes; temp_boxes];
                    cur_confidences = [cur_confidences; temp_confidence];
                    cur_image_ids = [cur_image_ids; {test_scenes(i).name}];
                end                
            end
        end
    end

    
    %cur_x_min = rand(15,1) * size(img,2);
    %cur_y_min = rand(15,1) * size(img,1);
    %cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    %cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]
    %cur_image_ids(1:15,1) = {test_scenes(i).name};
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end