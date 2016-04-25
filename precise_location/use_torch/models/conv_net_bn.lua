require 'nn'

local conv_net_bn = nn.Sequential()

local function ConvBNReLU(nInputePlane, nOutputPlane, nFilterSize)
    conv_net_bn:add(nn.SpatialConvolution(nInputePlane, nOutputPlane, nFilterSize, nFilterSize, 1, 1, (nFilterSize-1)/2, (nFilterSize-1)/2))
    conv_net_bn:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    conv_net_bn:add(nn.ReLU(true))
    return conv_net_bn
end

ConvBNReLU(3, 8, 7):add(nn.Dropout(0.3))
ConvBNReLU(8, 8, 7):add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(8, 16, 5):add(nn.Dropout(0.4))
ConvBNReLU(16, 16, 5):add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(16, 32, 3):add(nn.Dropout(0.4))
ConvBNReLU(32, 32, 3):add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())

conv_net_bn:add(nn.View(32*30*15))
conv_net_bn:add(nn.Linear(32*30*15, 16*30*15))
conv_net_bn:add(nn.BatchNormalization(16*30*15))
conv_net_bn:add(nn.ReLU(true))
conv_net_bn:add(nn.Linear(16*30*15, 4*30*15))
conv_net_bn:add(nn.BatchNormalization(4*30*15))
conv_net_bn:add(nn.ReLU(true))
conv_net_bn:add(nn.Linear(4*30*15, 30*15))
conv_net_bn:add(nn.BatchNormalization(30*15))
conv_net_bn:add(nn.ReLU(true))
conv_net_bn:add(nn.Linear(30*15, 8))

local function MSRinit(net)
    local function init(name)
        for k, v in pairs(net:findModules(name)) do
            local n = v.kW * v.kH * v.nOutputPlane
            v.weight:normal(0, math.sqrt(2/n))
            v.bias:zero()
        end
    end
    init'nn.SpatialConvolution'
end

MSRinit(conv_net_bn)

-- check that we can propagate forward without errors
-- should get 16x8 tensor
print(#conv_net_bn:forward(torch.Tensor(16, 3, 240, 120)))

return conv_net_bn