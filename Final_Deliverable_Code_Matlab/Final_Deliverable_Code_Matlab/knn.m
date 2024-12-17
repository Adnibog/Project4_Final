% Main Script for KNN Classification on PGM Images with Random Train/Test Split

% Parameters
imageFolder = uigetdir(pwd, 'Select the root folder of the image dataset');
numClassesToSelect = 35; % Number of classes to randomly select
numTrain = 8; % Number of training images per class
numTest = 2;  % Number of test images per class
kValue = 5; % Number of nearest neighbors

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
    
    % Assign random `numTrain` images to training and `numTest` images to testing
    trainIndices = shuffledIndices(1:numTrain); % Random training indices
    testIndices = shuffledIndices(numTrain + 1:numTrain + numTest); % Random test indices
    
    % Load and process training images
    for imgIdx = trainIndices
        imgPath = fullfile(classImages(imgIdx).folder, classImages(imgIdx).name);
        img = imread(imgPath);
        if size(img, 3) == 3
            img = rgb2gray(img); % Convert to grayscale if needed
        end
        if imageHeight == 0
            [imageHeight, imageWidth] = size(img); % Determine image size
        end
        img = imresize(img, [imageHeight, imageWidth]); % Resize to consistent dimensions
        imgVector = double(img(:)); % Vectorize the image
        
        % Add to training data
        trainData = [trainData, imgVector];
        trainLabels = [trainLabels; labelIdx];
    end
    
    % Load and process test images
    for imgIdx = testIndices
        imgPath = fullfile(classImages(imgIdx).folder, classImages(imgIdx).name);
        img = imread(imgPath);
        if size(img, 3) == 3
            img = rgb2gray(img); % Convert to grayscale if needed
        end
        img = imresize(img, [imageHeight, imageWidth]); % Resize to consistent dimensions
        imgVector = double(img(:)); % Vectorize the image
        
        % Add to test data
        testData = [testData, imgVector];
        testLabels = [testLabels; labelIdx];
    end
    
    fprintf('Loaded %d training and %d test images for class %d.\n', numTrain, numTest, labelIdx);
    labelIdx = labelIdx + 1;
end

% Train and Test KNN Classifier
fprintf('Training KNN classifier with k=%d...\n', kValue);

% Transpose data (rows = samples, columns = features)
trainData = trainData';
testData = testData';

% Use MATLAB's built-in knnsearch for KNN
mdl = fitcknn(trainData, trainLabels, 'NumNeighbors', kValue);

% Predict Test Labels
fprintf('Classifying test data using KNN...\n');
predictedLabels = predict(mdl, testData);

% Evaluate Classification Accuracy
accuracy = sum(predictedLabels == testLabels) / length(testLabels) * 100;
fprintf('KNN Classification Accuracy: %.2f%%\n', accuracy);

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