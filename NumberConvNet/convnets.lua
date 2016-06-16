#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"

-- Test neural network which determines digit from handwritten images of digits
-- Sample run on 1000 tests with a training rate of 1

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride)
cnn = nn.Sequential()
-- Input = 3*8*8 image, Output = 4*8*8 since the padding of 1 will ensure the same size remains
cnn:add(nn.SpatialConvolution(3,4,3,3,1,1,1,1))
cnn:add(nn.ReLU())
cnn:add(nn.Sigmoid())
-- Input = 4*8*8 image, Output = 1*8*8 since the padding of 1 will ensure the same size remains
cnn:add(nn.SpatialConvolution(4,1,3,3,1,1,1,1))
cnn:add(nn.ReLU())
cnn:add(nn.Sigmoid())
-- Reduce the size from 1*8*8 to 1*4*4 (Not done right now as the image is already small)
-- cnn:add(nn.SpatialMaxPooling(2,2,2,2))
-- Reshape from a 3d array to a 1d array
cnn:add(nn.Reshape(1*8*8))
-- Apply a completely connected neural network to get 10 outputs
cnn:add(nn.Linear(1*8*8,10))
cnn:add(nn.Sigmoid())

criterion = nn.MSECriterion()

for p=1,500 do
	for i=0,9 do
	    for j=1,15 do
	        local output = torch.Tensor(10);
	        imgfile = 'images/numbers/' .. i .. '.' .. j .. '.png'
	        local img = image.load(imgfile,3)
	        img:resize(3,8,8)
	        for k=1,10 do
	            output[k] = 0.0
	        end
	        output[i+1] = 1.0
	        -- Forward prop in the neural network
	        criterion:forward(cnn:forward(img), output)
	        -- Reset gradient accumulation
	        cnn:zeroGradParameters()
	        -- Accumulate gradients and back propogate
	        cnn:backward(img, criterion:backward(cnn.output, output))
	        -- Update with a learning rate
	        cnn:updateParameters(0.1)
	    end
	end
end

correct_matches = 0
accuracy = 0
total = 0

-- Test the network
for q=0,9 do
    imgfile = 'images/test_numbers/test' .. q .. '.png'
    print(imgfile)
    local img = image.load(imgfile,3)
    img:resize(3,8,8)
    results = cnn:forward(img)[1]
    best_result = torch.max(results)
    local cur_acc = results[q+1]
    total = 0
    for p=1,10 do
    	total = total + results[p]
    	if results[p] == best_result then
    		answer = p-1
    	end
    end
    print(answer)
    accuracy = accuracy + cur_acc / total
    if answer == q then
    	correct_matches = correct_matches + 1
    end
end

print('Final Results:')
print('Correct Matches = ' .. correct_matches)
print('Accuracy = ' .. accuracy)