% Main Script for SVM Classification on PGM Images with Randomly Selected 35 Classes

% Parameters
imageFolder = uigetdir(pwd, 'Select the root folder of the image dataset');
numClassesToSelect = 35; % Number of classes to randomly select
numTrain = 8; % Number of training images per class
numTest = 2;  % Number of test images per class
svmKernel = 'linear'; % SVM kernel type ('linear', 'rbf', etc.)

% Load PGM Images Recursively
fprintf('Loading PGM images from folder: %s\n', imageFolder);
imageFiles = dir(fullfile(imageFolder, '**', '*.pgm')); % Search for PGM files
if isempty(imageFiles)
    error('No PGM images found in the selected folder.');
end

% Organize Images by Class
% Each folder represents one class
classFolders = unique({imageFiles.folder}); % Get unique class folders
numAvailableClasses = length(classFolders);

fprintf('Found %d available classes.\n', numAvailableClasses);

% Check if there are enough classes
if numAvailableClasses < numClassesToSelect
    error('Not enough classes in the dataset. Required: %d, Found: %d.', numClassesToSelect, numAvailableClasses);
end

% Randomly select 35 classes
rng('shuffle'); % Seed random generator for randomness
selectedClassIndices = randperm(numAvailableClasses, numClassesToSelect);
selectedClassFolders = classFolders(selectedClassIndices);

fprintf('Randomly selected %d classes.\n', numClassesToSelect);

% Initialize variables
imageHeight = 0; 
imageWidth = 0;
trainData = [];
trainLabels = [];
testData = [];
testLabels = [];

labelIdx = 1;

% Load Images, Split into Training and Testing
for classIdx = 1:numClassesToSelect
    folderPath = selectedClassFolders{classIdx};
    classImages = dir(fullfile(folderPath, '*.pgm')); % Images for this class
    
    if length(classImages) < numTrain + numTest
        error('Not enough images in class folder: %s. Required: %d.', folderPath, numTrain + numTest);
    end
    
    % Shuffle the images to randomize selection
    shuffledIndices = randperm(length(classImages));
    
    % Load and resize images
    for imgIdx = 1:(numTrain + numTest)
        imgPath = fullfile(classImages(shuffledIndices(imgIdx)).folder, classImages(shuffledIndices(imgIdx)).name);
        img = imread(imgPath);
        if size(img, 3) == 3
            img = rgb2gray(img); % Convert to grayscale if needed
        end
        if imageHeight == 0
            [imageHeight, imageWidth] = size(img); % Determine image size
        end
        img = imresize(img, [imageHeight, imageWidth]); % Resize to consistent dimensions
        imgVector = double(img(:)); % Vectorize the image
        
        % Split into training and test data
        if imgIdx <= numTrain
            trainData = [trainData, imgVector]; % Training data
            trainLabels = [trainLabels; labelIdx]; % Training labels
        else
            testData = [testData, imgVector]; % Test data
            testLabels = [testLabels; labelIdx]; % Test labels
        end
    end
    fprintf('Loaded %d training and %d test images for class %d.\n', numTrain, numTest, labelIdx);
    labelIdx = labelIdx + 1;
end

% Train SVM Classifier
fprintf('Training SVM classifier with %s kernel...\n', svmKernel);

% Transpose data (rows = samples, columns = features)
trainData = trainData';
testData = testData';

% Use MATLAB's built-in fitcecoc for multi-class SVM
svmModel = fitcecoc(trainData, trainLabels, 'Learners', templateSVM('KernelFunction', svmKernel));

% Predict Test Labels
fprintf('Classifying test data using SVM...\n');
predictedLabels = predict(svmModel, testData);

% Evaluate Classification Accuracy
accuracy = sum(predictedLabels == testLabels) / length(testLabels) * 100;
fprintf('SVM Classification Accuracy: %.2f%%\n', accuracy);

% Display Test Images and Predictions
figure;
numDisplay = min(10, size(testData, 1)); % Display up to 10 test images
for i = 1:numDisplay
    subplot(2, numDisplay, i);
    testImage = reshape(testData(i, :)', imageHeight, imageWidth);
    imshow(uint8(testImage));
    title(sprintf('True: %d', testLabels(i)));
    
    subplot(2, numDisplay, i + numDisplay);
    imshow(uint8(testImage));
    title(sprintf('Predicted: %d', predictedLabels(i)));
end
sgtitle('Test Images: True vs Predicted Labels');