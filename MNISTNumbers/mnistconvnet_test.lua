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

correct_matches = 0
accuracy = 0
total = 0

cnn = torch.load('model.torch')

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

no_of_tests = 10000
-- Test the network
for q=1,no_of_tests do
	local example = testset[q]
	local input = example.x
	local output_val = example.y
    input:resize(1,28,28)
    input = input:double()
    input = input / 255.0
    results = cnn:forward(input)
    best_result = torch.max(results)
    local cur_acc = results[output_val+1]
    total = 0
    for p=1,10 do
    	total = total + results[p]
    	if results[p] == best_result then
    		answer = p-1
    	end
    end
    accuracy = accuracy + cur_acc / total
    if answer == output_val then
    	correct_matches = correct_matches + 1
    end
    print(q .. ' => ' .. output_val .. '|' .. answer)
end
print('Final Results:')
print('Correct Matches = ' .. correct_matches .. '/' .. no_of_tests)
print('Accuracy = ' .. accuracy / no_of_tests)