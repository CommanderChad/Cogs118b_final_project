
imageFolder = 'C:\Users\chads\Downloads\alligator';


imageFiles = dir(fullfile(imageFolder, '*.jpg'));
numImages = length(imageFiles);  % should be 27 for bees

imgHeight = 60;
imgWidth  = 60;
allImages = zeros(imgHeight, imgWidth, numImages);

for k = 1:numImages
    img = imread(fullfile(imageFolder, imageFiles(k).name));
    
    % Convert to grayscale 
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    img = imresize(img, [imgHeight imgWidth]);
    
    img = im2double(img);
    
    % Store in array
    allImages(:,:,k) = img;
end

%flatten each image
allImagesFlattened = reshape(allImages, imgHeight*imgWidth, numImages);

% Save to .mat file
save('alligator_images_60x60.mat', 'allImages', 'allImagesFlattened');

