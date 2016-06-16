#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
mnist = require "mnist"

-- Test neural network which determines digit from handwritten images of digits
-- Sample run on 1000 tests with a training rate of 0.3

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
cnn = nn.Sequential()
-- Input = 1*28*28 image, Output = 32*28*28
cnn:add(nn.SpatialConvolution(1,32,5,5,1,1,2,2))
cnn:add(nn.Tanh())
cnn:add(nn.SpatialMaxPooling(3,3,3,3,1,1))
-- Input = 32*10*10 image, Output = 64*10*10
cnn:add(nn.SpatialConvolution(32,64,5,5,1,1,2,2))
cnn:add(nn.Tanh())
-- Input = 64*10*10 image, Output = 64*5*5
cnn:add(nn.SpatialMaxPooling(2,2,2,2))
-- Reshape from a 3d array to a 1d array
cnn:add(nn.Reshape(64*5*5))
-- Apply a completely connected neural network
cnn:add(nn.Linear(64*5*5,256))
cnn:add(nn.Tanh())
-- Another completely connected neural network to get 10 outputs
cnn:add(nn.Linear(256,10))
cnn:add(nn.Tanh())

criterion = nn.MSECriterion()

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

no_of_training_cases = 60000

for p=1,no_of_training_cases do
	print(p)
	local example = trainset[p]
	local input = example.x
	local output_val = example.y
    local output = torch.Tensor(10);
    input:resize(1,28,28)
    input = input:double()
   	input = input / 255.0
    for k=1,10 do
        output[k] = 0.0
    end
    output[output_val+1] = 1.0
    -- Forward prop in the neural network
    criterion:forward(cnn:forward(input), output)
    -- Reset gradient accumulation
    cnn:zeroGradParameters()
    -- Accumulate gradients and back propogate
    cnn:backward(input, criterion:backward(cnn.output, output))
    -- Update with a learning rate
    cnn:updateParameters(0.3)
end

print('Done training, saving model if needed')
torch.save('model.torch', cnn)