%close all
%clear
%run('vlfeat-0.9.21/toolbox/vl_setup')

data_path = 'LabHOG/data/'; %change if you want to work with a network copy
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes

load('w_pos_6713_neg_100010_lamba_0_0001')
load('b_pos_6713_neg_100010_lamba_0_0001')

[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);

[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)