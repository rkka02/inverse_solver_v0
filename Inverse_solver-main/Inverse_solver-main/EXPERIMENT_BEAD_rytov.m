clc, clear;
cd0 = matlab.desktop.editor.getActiveFilename;

cd0 = cd0(1:strfind(cd0,'EXPERIMENT_BEAD.m')-2);
addpath(genpath(cd0));
used_gpu_device=1;
gpu_device=gpuDevice(used_gpu_device);
addpath(genpath('C:\rkka_Projects\inverse_solver_v0\Inverse_solver-main\Inverse_solver-main\Codes'));
%% set the experimental parameters
cddata = fullfile(cd0, 'C:\rkka_Projects\inverse_solver_v0\Inverse_solver-main\Inverse_solver-main\Data');
bg_file = fullfile(cddata, 'background.tif');
sp_file = fullfile(cddata, 'sample.tif');

cd(cddata)

%1 optical parameters
MULTI_GPU=false;

params=BASIC_OPTICAL_PARAMETER();
params.NA=1.2;
params.RI_bg=1.3355;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=false;true;
params.size=[0 0 151]; 
params.use_GPU = true;

%2 illumination parameters
field_retrieval_params=FIELD_EXPERIMENTAL_RETRIEVAL.get_default_parameters(params);
field_retrieval_params.resolution_image=[1 1]*0.082;%[1 1]*(5.5/100);
field_retrieval_params.conjugate_field=true;
field_retrieval_params.use_abbe_correction=true;

% 1. Aberration correction
field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);

% Aberration correction data
[input_field,field_trans,params]=field_retrieval.get_fields(bg_file,sp_file);

% Display results: transmitted field
figure;orthosliceViewer(squeeze(abs(field_trans(:,:,:)./input_field(:,:,:))),'displayrange',[0 2]); colormap gray; title('Amplitude')
figure;orthosliceViewer(squeeze(angle(field_trans(:,:,:)./input_field(:,:,:)))); colormap jet; title('Phase')

%% solve with rytov
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
RI_rytov=rytov_solver.solve(input_field,field_trans);
figure;orthosliceViewer(real(RI_rytov)); title('Rytov')
