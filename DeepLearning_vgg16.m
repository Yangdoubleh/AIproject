%Alexnet setting
net = vgg19;
camera = webcam('Integrated Webcam');
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net);

%DB setting / Training Data : ValidationData = 6:4
imds = imageDatastore('C:\Users\user\Desktop\종설프\DB-224,224','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.6,'randomized');
numTrainimages = numel(imdsTrain.Labels);
numClasses = numel(categories(imds.Labels));
idx = randperm(numTrainimages, 16);
figure
for i = 1:16
    subplot(4,4,i)
    l = readimage(imdsTrain,idx(i));
    imshow(l)
end

%Transfer Learning
layersTransfer = net.Layers(1:end-3);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%Image Augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, 'DataAugmentation', imageAugmenter);

%Training option
options = trainingOptions('sgdm',...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 7, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch',...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


netTransfer = trainNetwork(augimdsTrain, layers, options);

Ypredicted = classify(netTransfer,imds);
plotconfusion(imds.Labels,Ypredicted)

%live Webcam
a = figure
a.Position(3) = 2*a.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
ax2.ActivePositionProperty = 'position';

while ishandle(a)
    %ax1
    lc = snapshot(camera);
    image(ax1,lc)
    lc = imresize(lc, inputSize(1:2));
    [label, score] = classify (netTransfer, lc);
    title(ax1,{char(label),num2str(max(score),2)});
    
    %index score
   [~,lcidx] = sort(score,'descend');
   lcidx = lcidx(7:-1:1);
   classes = netTransfer.Layers(end).Classes;
   scoreTop = score(lcidx);
   classNamesTop = string(classes(lcidx));
   
   %ax2
   barh(ax2,scoreTop)
   title(ax2,'Top score')
   xlabel(ax2,'Probability')
   xlim(ax2,[0 1])
   yticklabels(ax2,classNamesTop)
   ax2.YAxisLocation = 'right';
   
   drawnow
end
