% Main Script for PCA on PGM Images in Recursive Folders

% Parameters
imageFolder = uigetdir(pwd, 'Select the root folder of the image dataset');
numComponents = 400; % Number of principal components for reconstruction
pcaType = 1; % 1 for Eigen decomposition, 2 for SVD

% Load PGM Images Recursively
fprintf('Loading PGM images from folder: %s\n', imageFolder);
imageFiles = dir(fullfile(imageFolder, '**', '*.pgm')); % Adjust extension for .pgm files
if isempty(imageFiles)
    error('No images found');
end

% Read the first image to get dimensions
sampleImage = imread(fullfile(imageFiles(1).folder, imageFiles(1).name));
[imageHeight, imageWidth] = size(sampleImage);

% Initialize matrix to store all images
numImages = length(imageFiles);
imageMatrix = zeros(imageHeight * imageWidth, numImages);

% Load and vectorize all images
for i = 1:numImages
    imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    img = imread(imgPath);
    if size(img, 3) == 3
    end
    imageMatrix(:, i) = double(img(:)); % Vectorize and store
end
fprintf('Loaded %d PGM images of size %dx%d.\n', numImages, imageHeight, imageWidth);

% Perform PCA
fprintf('Performing PCA on the dataset...\n');
[P, s, X_new, per] = PCA_C(imageMatrix, pcaType);

% Visualize Results
% Display the mean image
meanImage = mean(imageMatrix, 2);
figure;
imshow(reshape(meanImage, imageHeight, imageWidth), []);
title('Mean Image');

% Display the top eigenfaces
numEigenfaces = min(10, size(P, 2));
figure;
for i = 1:numEigenfaces
    subplot(2, 5, i);
    imshow(reshape(P(:, i), imageHeight, imageWidth), []);
    title(sprintf('Eigenface %d', i));
end
sgtitle('Top Eigenfaces');

% Reconstruct Images Using Top Principal Components
fprintf('Reconstructing images using %d principal components...\n', numComponents);
meanImage = mean(imageMatrix, 2); % Mean image
topComponents = P(:, 1:numComponents); % Top principal components
reconstructedMatrix = topComponents * X_new(1:numComponents, :) + meanImage;

% Select 10 Random Images for Display
numDisplay = min(10, numImages); % Number of images to display
randomIndices = randperm(numImages, numDisplay); % Get 10 random unique indices

% Display Original and Reconstructed Images
figure;
for i = 1:numDisplay
    % Original image
    subplot(2, numDisplay, i);
    originalImage = reshape(imageMatrix(:, randomIndices(i)), imageHeight, imageWidth);
    imshow(uint8(originalImage));
    title(sprintf('Original %d', randomIndices(i)));
    
    % Reconstructed image
    subplot(2, numDisplay, i + numDisplay);
    reconstructedImage = reshape(reconstructedMatrix(:, randomIndices(i)), imageHeight, imageWidth);
    imshow(uint8(reconstructedImage));
    title(sprintf('Reconstructed %d', randomIndices(i)));
end
sgtitle(sprintf('Original vs Reconstructed Images (Top %d Components)', numComponents));