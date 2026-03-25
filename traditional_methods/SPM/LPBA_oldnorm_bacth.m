% =====================================================================================
% SPM12 Automated Batch Script: Old Normalise (Estimate & Separate Write) for LPBA40
% ===================================================================================

% 1. Initialize SPM Engine
spm('defaults', 'FMRI');
spm_jobman('initcfg');

% 2. Define the dataset directory and the number of subjects
base_dir = 'path/to/LPBA40';
num_pairs = 90;
process_times = NaN(num_pairs, 1);

fprintf('A total of %d data pairs are scheduled for processing. Starting batch processing...\n', num_pairs);

% 3. Begin looping through each test registration pair
for i = 1:num_pairs
    current_pair_dir = fullfile(base_dir, sprintf('pairs%d', i));
    src_img_path   = fullfile(current_pair_dir, 'x.nii');
    tar_img_path   = fullfile(current_pair_dir, 'y.nii');
    src_label_path = fullfile(current_pair_dir, 'x_seg.nii');
    
    if ~exist(src_img_path, 'file') || ~exist(tar_img_path, 'file') || ~exist(src_label_path, 'file')
        fprintf('[Warning] Data for pairs%d is incomplete; skipped!\n', i);
        continue; 
    end

    src_img   = [src_img_path, ',1'];
    tar_img   = [tar_img_path, ',1'];
    src_label = [src_label_path, ',1'];
    
    sn_mat_path = strrep(src_img_path, '.nii', '_sn.mat');
    
    fprintf('Processing pair %d/%d...\n', i, num_pairs);
    
    % Clear the batch configuration from the previous iteration
    clear matlabbatch; 
    
    % ---------------------------------------------------------------------
    % Module 1: Estimate (Calculating Nonlinear Deformation Field Parameters)
    % ---------------------------------------------------------------------
    matlabbatch{1}.spm.tools.oldnorm.est.subj.source = {src_img};
    matlabbatch{1}.spm.tools.oldnorm.est.subj.wtsrc = '';
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.template = {tar_img};
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.weight = '';
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.smosrc = 6;
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.smoref = 6;
    % subj none mni
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.regtype = 'none'; 
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.cutoff =  15;
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.nits = 30;
    matlabbatch{1}.spm.tools.oldnorm.est.eoptions.reg = 1;

    % ---------------------------------------------------------------------
    % Module 2: Write - warpping moving image (Trilinear)
    % ---------------------------------------------------------------------
    matlabbatch{2}.spm.tools.oldnorm.write.subj.matname = {sn_mat_path};
    matlabbatch{2}.spm.tools.oldnorm.write.subj.resample = {src_img};
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.preserve = 0;
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.vox = [NaN NaN NaN];
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.interp = 1; % Trilinear
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.wrap = [0 0 0];
    matlabbatch{2}.spm.tools.oldnorm.write.roptions.prefix = 'w_';

    % ---------------------------------------------------------------------
    % Module 3: Write - warpping label (Nearest neighbour)
    % ---------------------------------------------------------------------
    matlabbatch{3}.spm.tools.oldnorm.write.subj.matname = {sn_mat_path};
    matlabbatch{3}.spm.tools.oldnorm.write.subj.resample = {src_label};
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.preserve = 0;
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.vox = [NaN NaN NaN];
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.interp = 0; % Nearest neighbour
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.wrap = [0 0 0];
    matlabbatch{3}.spm.tools.oldnorm.write.roptions.prefix = 'w_';

    % 4. Submit to the SPM backend engine for execution and performance testing
    try
        tic; 
        spm_jobman('run', matlabbatch);
        process_times(i) = toc; 
        fprintf('The %dth pair has been successfully processed! Time taken: %.2f seconds\n\n', i, process_times(i));
    catch ME
        fprintf('The %dth pair failed! Error message: %s\n\n', i, ME.message);
    end
    
end

% =========================================================================
% 5. Output Analysis Results
% =========================================================================
valid_times = process_times(~isnan(process_times));
num_valid = length(valid_times);
fprintf('\n======================================================\n');
if num_valid > 0
    mean_time = mean(valid_times);
    var_time = var(valid_times);
    std_time = std(valid_times);
    
    fprintf('Batch Processing Statistics Report:\n');
    fprintf('Number of successfully processed pairs: %d / %d pair\n', num_valid, num_pairs);
    fprintf('Average time per pair: %.2f seconds\n', mean_time);
    fprintf('Variance in time taken per pair: %.2f (seconds^2)\n', var_time);
    fprintf('Standard deviation in time per pair: %.2f seconds\n', std_time);
    fprintf('Total elapsed time: %.2f minutes\n', sum(valid_times)/60);
else
    fprintf('No data was processed successfully; time statistics cannot be calculated.\n');
end
fprintf('======================================================\n');
