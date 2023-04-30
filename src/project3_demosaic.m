% Abd Elelah Arafah 400197623
% This algorithm follows a machine learning-based linear regression 
% approach for demosaicing an image. 
clc;
clear; 
close all; 

% input the raw image. Change this to the image you want to test
input_image = imread('sample_image_1.png');

%%%%% 1/3 comment this out if ground truth image does not exist %%%%%%
%%%%% Only uncomment if you have an original rgb image to compare the output %%%%%%
% Ground truth demosiaced image for comparison. 
% Change this to the ground truth image of the input image.
gt_image = imread('gt_image_1.png');
%%%%% 1/3 %%%%%

% Learning from 3 different HR images from the DIV2K database
learn_img_1 = imresize(imread('0804.png'), [512,512]);
learn_img_2 = imresize(imread('0805.png'), [512,512]);
learn_img_3 = imresize(imread('0806.png'), [512,512]);

% Combine learn images into one learn images for more accurate learning. use concatenate function
combined_learning_img = cat(2, learn_img_1, learn_img_2, learn_img_3);

% display input image
figure;
imshow(input_image);
title('Raw Data Input Image');

% get dimensions
[row_count_input, col_count_input, layers] = size(input_image);
[row_count_learn, col_count_learn, layers2] = size(combined_learning_img);

% Extract bayer filter
image_bayer_var = image_bayer(input_image);
learn_bayer_img = image_bayer(combined_learning_img);

%output original learning image
figure
imshow(combined_learning_img)
title('Image for training');

% bayer image
figure
imshow(image_bayer_var)
title('Bayer pattern image');

% change quantities into double
train_image_double = double(combined_learning_img);
learn_bayer_img = double(learn_bayer_img);
learn_layer_img = sum(learn_bayer_img, 3); % add it in single img

% expand borders with custom function
learn_img_exp_borders = extendImageBorders(learn_layer_img);

% Training with machine learning-based linear regression for demosaicing
% Initialize indices for different patch types
redIdx = 0;
green1Idx = 0;
green2Idx = 0;
blueIdx = 0;

% Define input matrices for each tile type
tileCount = (row_count_learn*col_count_learn/4);
inputMatrix = zeros(tileCount,49,4);

%preallocate y-matrix (true values)
outputMatrix = zeros(tileCount,8);

% Calculate input matrices for each tile type
% Scan each pixel of learning image (offset by 3 pixels to get center pixel of expanded image)
for row = 4:(row_count_learn + 3)
    for col = 4:(col_count_learn + 3)
        % Simulate 7x7 patches
        subMatrix = learn_img_exp_borders(row-3:row+3,col-3:col+3); 
        % Convert submatrix to a row vector
        subMatrix = reshape(subMatrix.',1,[]);

        % Update input and output matrices based on the Bayer filter pattern
        if (mod(row,2)==0 && mod(col,2)==0)
            % Red tile (even row, even column)
            redIdx=redIdx+1;
            inputMatrix( redIdx, :, 1 ) = subMatrix;
            outputMatrix(redIdx,1)=train_image_double(row-3,col-3,2);%green1
            outputMatrix(redIdx,2)=train_image_double(row-3,col-3,3);%blue1
        elseif mod(row,2)==0 && mod(col,2)==1
            % Green tile - first row (even row, odd column)
            green1Idx=green1Idx+1;
            inputMatrix( green1Idx, :, 2 ) = subMatrix;
            outputMatrix(green1Idx,3)=train_image_double(row-3,col-3,1);%red1
            outputMatrix(green1Idx,4)=train_image_double(row-3,col-3,3);%blue2
        elseif mod(row,2)==1 && mod(col,2)==0
            % Green tile - second row (odd row, even column)
            green2Idx=green2Idx+1;
            inputMatrix( green2Idx, :, 3 ) = subMatrix;
            outputMatrix(green2Idx,5)=train_image_double(row-3,col-3,1);%red2
            outputMatrix(green2Idx,6)=train_image_double(row-3,col-3,3);%blue3
        else
            % Blue tile (odd row, odd column)
            blueIdx=blueIdx+1;
            inputMatrix( blueIdx, :, 4 ) = subMatrix;
            outputMatrix(blueIdx,7)=train_image_double(row-3,col-3,1);%red3
            outputMatrix(blueIdx,8)=train_image_double(row-3,col-3,2);%green2
        end
    end
end

% Compute linear regression coefficients for each color channel (8 optimal coefficient matrices)
regressionCoefficients = zeros(49,8);
for idx = 1:8
    % Select the appropriate input matrix
    inputMatrixIdx = round(idx/2); % Adjusts index range to 1-4
    currentInputMatrix = inputMatrix(:,:,inputMatrixIdx);
    transposedInputMatrix = transpose(currentInputMatrix);
    
    % Calculate the linear regression coefficients
    regressionCoefficients(:,idx) = inv(transposedInputMatrix*currentInputMatrix)*transposedInputMatrix*outputMatrix(:,idx);
end

disp(['Linear Regression model completed!' newline 'Now preforming Demosaicing...']);

% Performing demosaicing
% format image the image
bayerImageDouble = double(image_bayer_var);
demosaicedImage = bayerImageDouble; 
bayerImageSingleLayer = sum(bayerImageDouble,3);

% Expand borders for calculation purposes
expandedBayerImage = extendImageBorders(bayerImageSingleLayer);

% Compute demosaic image
offset=-3;
for row = 4:(row_count_input + 3)
    for col = 4:(col_count_input + 3)
        % Extract 7x7 patches
        subMatrix = expandedBayerImage(row-3:row+3,col-3:col+3);
        subMatrix=reshape(subMatrix.',1,[]);

        % Update demosaicedImage based on the Bayer filter pattern
        if (mod(row,2)==0 && mod(col,2)==0)
            % Red tile (even row, even column)
            demosaicedImage(row+offset,col+offset,2)=subMatrix*regressionCoefficients(:,1);
            demosaicedImage(row+offset,col+offset,3)=subMatrix*regressionCoefficients(:,2);
            
        elseif mod(row,2)==0 && mod(col,2)==1
            % Green tile - first row (even row, odd column)
            demosaicedImage(row+offset,col+offset,1)=subMatrix*regressionCoefficients(:,3);
            demosaicedImage(row+offset,col+offset,3)=subMatrix*regressionCoefficients(:,4);
        elseif mod(row,2)==1 && mod(col,2)==0
            % Green tile - second row (odd row, even column)
            demosaicedImage(row+offset,col+offset,1)=subMatrix*regressionCoefficients(:,5);
            demosaicedImage(row+offset,col+offset,3)=subMatrix*regressionCoefficients(:,6);
        else
            % Blue tile (odd row, odd column)
            demosaicedImage(row+offset,col+offset,1)=subMatrix*regressionCoefficients(:,7);
            demosaicedImage(row+offset,col+offset,2)=subMatrix*regressionCoefficients(:,8);
        end
    end
end

% Computing errors
% Convert back to uint8
demosaicedImage=uint8(demosaicedImage);

%%%%% 2/3 comment this out if ground truth image does not exist %%%%%%
%%%%% Only uncomment if you have an original rgb image to compare the output %%%%%%
% Display gt image
figure;
imshow(gt_image);
title('Ground truth Image for comparison');
%%%%% 2/3 %%%%%

% Matlab builtin image demosiacing
% specify the Bayer pattern of the raw image (RGGB, GRBG, etc.)
bayer_patterns_mosiac = {'rggb', 'grbg', 'gbrg', 'bggr'};
%input_image = im2gray(input_image);
% demosaic the raw image using the built-in demosaic function
rgb_image = demosaic(input_image, bayer_patterns_mosiac{1});

% display the demosaicked RGB image from the builtin function
figure;
imshow([rgb_image, demosaicedImage]);
title('Matlab built-in vs Linear regression Image');

%%%%% 3/3 comment this out if ground truth image does not exist %%%%%%
%%%%% Only uncomment if you have an original rgb image to compare the output %%%%%%
%RMSE compute
RMSE_error_of_output = immse(demosaicedImage,gt_image)
RMSE_error_matlab_builtin = immse(rgb_image,gt_image)
%%%%% 3/3 %%%%%

% Custom functions
% Convert input image to a Bayer image
function outputImage = image_bayer(inputImage)
    % Initialize Bayer filter to the size of the input image
    [numRows, numCols, ~] = size(inputImage);
    bayerFilterRGB = inputImage * 0; % Preallocate filter

    % Generate Bayer mosaic patterns in RGB format
    % The same uint8 format as the input image
    for row = 1:numRows
        for col = 1:numCols
            % Red
            if mod(row, 2) == 1 && mod(col, 2) == 1
                bayerFilterRGB(row, col, 1) = 255;
            % Blue
            elseif mod(row, 2) == 0 && mod(col, 2) == 0
                bayerFilterRGB(row, col, 3) = 255;
            % Green
            else
                bayerFilterRGB(row, col, 2) = 255;
            end
        end
    end
    outputImage = inputImage .* (bayerFilterRGB / 255);
end

% Extend image borders by mirroring neighboring pixels
% Extends by 3 pixels on each side of the original image
function extendedImage = extendImageBorders(inputImage)
    [numRows, numCols, ~] = size(inputImage);
    
    % Preallocate image matrix to extended size and copy original with offset
    extendedImage = zeros(numRows + 6, numCols + 6);
    extendedImage(4:numRows + 3, 4:numCols + 3) = inputImage;

    % Add mirrored elements
    % Left and right mirror components
    for col = 1:3
        extendedImage(4:numRows + 3, col:col) = inputImage(1:numRows, 5 - col);
        extendedImage(4:numRows + 3, col + numCols + 3:col + numCols + 3) = inputImage(1:numRows, numCols - col);
    end

    % Top and bottom mirror components
    for row = 1:3
        extendedImage(row:row, 1:numCols + 6) = extendedImage(8 - row:8 - row, 1:numCols + 6);
        extendedImage(row + numRows + 3:row + numRows + 3, 1:numCols + 6) = extendedImage(3 + numRows - row:3 + numRows - row, 1:numCols + 6);
    end
end
