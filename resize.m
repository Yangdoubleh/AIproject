%network setting
realnet = alexnet;
alinputSize = realnet.Layers(1).InputSize(1:2);
revgg = vgg16;
vgginputSize = revgg.Layers(1).InputSize(1:2);
regoogle = googlenet;
googleinputSize = regoogle.Layers(1).InputSize(1:2);

%resize code
raw = imageDatastore('C:\Users\user\Desktop\종설프\DB-227,227\Vinyl')
for m = 1:size(raw.Files)
    reimg = imread(raw.Files{m});
    resized_img = imresize(reimg, vgginputSize);%inputsize change
    imwrite(resized_img,sprintf('resized_img%03d.png',m));
end
