#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"

-- Test neural network which calculates the sum of the inputs
-- Sample run on 10000 tests with a training rate of 0.001

mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25))
-- Tangent transfer function
mlp:add(nn.Tanh())
mlp:add(nn.Linear(25, 1))

criterion = nn.MSECriterion()

for i=1,10000 do
    local input = torch.rand(10);
    local output = torch.Tensor(1);
    -- print(input)
    output[1] = 0
    for j=1,10 do
        output[1] = output[1] + input[j]
    end
    -- print(output)
    -- Forward prop in the neural network
    criterion:forward(mlp:forward(input), output)
    -- Reset gradient accumulation
    mlp:zeroGradParameters()
    -- Accumulate gradients and back propogate
    mlp:backward(input, criterion:backward(mlp.output, output))
    -- Update with a learning rate
    mlp:updateParameters(0.001)
end


-- Test the network
x = torch.Tensor(10)
x1 = torch.Tensor(10)
x2 = torch.Tensor(10)
for i=1,10 do
    x[i] = 0.2
    x1[i] = 0.3
    x2[i] = 0.4
end

print("Test 1:")
print(x)
print(mlp:forward(x))
print("Test 2:")
print(x1)
print(mlp:forward(x1))
print("Test 3:")
print(x2)
print(mlp:forward(x2))
