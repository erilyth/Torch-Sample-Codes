#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"

-- Test neural network which calculates the sum of the inputs
-- Sample run on 1000 tests with a training rate of 0.01

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride)
cnn = nn.Sequential()
-- Input = 3*8*8 image, Output = 4*8*8 since the padding of 2 will ensure the same size remains
cnn:add(nn.SpatialConvolution(3,4,5,5,1,1,2,2))
cnn:add(nn.ReLU())
-- Reduce the size from 4*8*8 to 4*4*4
cnn:add(nn.SpatialMaxPooling(2,2,2,2))
-- Input = 4*4*4 image, Output = 5*4*4
cnn:add(nn.SpatialConvolution(4,5,5,5,1,1,2,2))
cnn:add(nn.ReLU())
-- Reduce the size from 5*2*2 to 5*2*2
cnn:add(nn.SpatialMaxPooling(2,2,2,2))
-- Convert into a linear 1D array
cnn:add(nn.Reshape(5*2*2))
-- Apply a completely connected neural network to get 10 outputs
cnn:add(nn.Linear(5*2*2,10))
cnn:add(nn.Sigmoid())
-- Get a number output (Another fully connected nerual net)
cnn:add(nn.Linear(10,10))
cnn:add(nn.Sigmoid())

criterion = nn.MSECriterion()

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
        cnn:updateParameters(0.01)
    end
end



-- Test the network
for i=1,15 do
    imgfile = 'images/numbers/' .. '1' .. '.' .. i .. '.png'
    print(imgfile)
    local img = image.load(imgfile,3)
    img:resize(3,8,8)
    print(cnn:forward(img))
end
