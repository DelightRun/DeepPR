require 'nn'
require 'cunn'
require 'cudnn'
local nninit = require 'nninit'

local cfg = {32, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 256, 'M'}

local features = nn.Sequential()

local nInputPlanes = 3

local width = 448
local height = 224

do
    for k, v in ipairs(cfg) do
        if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
            width = width / 2
            height = height / 2
        else
            local nOutputPlanes = v
            features:add(nn.SpatialConvolution(nInputPlanes, nOutputPlanes, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming))
            features:add(nn.SpatialBatchNormalization(nOutputPlanes))
            features:add(nn.ReLU())
            nInputPlanes = nOutputPlanes
        end
    end
end

local regressor = nn.Sequential()
regressor:add(nn.View(nInputPlanes*width*height))
while nInputPlanes ~= 1 do
    local nOutputPlanes = nInputPlanes / 4
    regressor:add(nn.Linear(nInputPlanes*width*height, nOutputPlanes*width*height))
    regressor:add(nn.BatchNormalization(nOutputPlanes*width*height))
    regressor:add(nn.ReLU())
    nInputPlanes = nOutputPlanes
end
regressor:add(nn.Linear(width*height, 8))

local model = nn.Sequential()
model:add(features):add(regressor)

model:cuda()
cudnn.convert(model, cudnn)

print(model)

model:clearState()
torch.save('vgg.t7', model)
