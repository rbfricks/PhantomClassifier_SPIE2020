%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% matWrapper.m
%
% DESCRIPTION:
% Demonstration script for importing a model trained with Tensorflow-Keras
% and using it for classification. Requires deep learning import packages
% from add-on explorer in MATLAB, and deep learning toolbox. Recommended
% to use MATLAB 2019b or higher.
%
% Produced for Mercury Phantom Detector
%
% Rafael Fricks
% Created October 1, 2019
% Editted October 3, 2019
% RAI Labs, Duke University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('UsefulTools')
pathName = 'E20';
modelName = 'VGG19';
% modelName = 'InceptionV3';

delay = 1; % delay in seconds between displaying slices

disp('Reading CT Series...')
[I, info] = readCTSeries();
n_slices = size(I,3);

%%% scroll range %%%
startSlice = 1; %which slice, if doing individual slices
endSlice = n_slices;

%%% path variables %%%
modelfile = [pathName '/' modelName '_architecture.json'];
weights = [pathName '/' modelName '_modelWeight.h5'];
fullModel = [pathName '/' modelName '_fullModel.h5']; %% not used in current version

numNames = {'1','2','3','4','5'};
catNames = {'Airgap or Partial Volume','NPS','Outside Phantom','TTF','Tapered Section'};

%%% load the desired model
disp('Importing the Model...')
warning('off','all') % the import utilities are still pretty rough
net = importKerasNetwork(modelfile,'WeightFile',weights,'OutputLayerType','classification');
% % net = importKerasNetwork(fullModel, 'OutputLayerType','classification'
warning('on','all')

disp('Displaying Series in Sequence with Predictions...')
ttp = zeros(endSlice-startSlice+1,1); %time to predict
for i=startSlice:endSlice %scroll through all slices
    %%% correct for end slices %%%
    prev = i - 1;
    if(prev==0)
        prev = 1;
    end
    next = i + 1;
    if(next>size(I,3))
        next = size(I,3);
    end
    
    %%% Begin image preprocessing %%%
    % DO NOT ADJUST: these window limits and preprocessing must match
    % input preprocessing during training, minus augmentation
    wmin = -1024;
    wmax = 1187;
    
    img = cat(3,I(:,:,prev), I(:,:,i), I(:,:,next));
    
    img = imresize(img,[256 256]);
    img(img>wmax) = wmax;
    img(img<wmin) = wmin;
    
    img = uint8(((img-wmin)*255)/(wmax-wmin));
    %%% End image preprocessing %%%
    
    %%% classify, then convert labels to readable strings %%%
    tic
    label = classify(net,img);
    ttp(i) = toc;
    
    k = renamecats(label,catNames);
    
    %%% Display prediction set %%%
    figure(1)
    subplot(131)
    imshow(img(:,:,1), [0 255])
    title('Previous Slice')
    xlabel(['Model Name: ' modelName])
    
    subplot(132)
    imshow(img(:,:,2), [0 255])
    title(['Slice #' num2str(i)])
    xlabel(['Predicition: ' char(k)])
    
    subplot(133)
    imshow(img(:,:,3), [0 255])
    title('Next Slice')
    pause(delay)
    
end

%%% show the network architecture of the deep learning model
figure(2)
plot(net)
view(270, 90)
title([modelName ' Classification Network'])
disp('Demo complete.')

%%% plot timing distribution of prediction times 
figure(3)
histogram(ttp(2:end),100,'normalization','probability') %first time discarded
title('Histogram (Empirical PMF) of Prediction Times')
ylabel('Probability')
xlabel(['Prediction Time (s, avg = ' num2str(mean(ttp(2:end))) ' s)'])

