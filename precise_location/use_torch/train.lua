require 'xlua'
require 'optim'
require 'cunn'
require 'provider'

local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save               (default "logs")            subdirectory to save logs
    -b,--batchSize          (default 50)                batch size
    -r,--learningRate       (default 0.001)             learning rate
    --epoch_step            (default 25)                epoch step
    --model                 (default conv_net_bn)       model name
    --max_epoch             (default 150)               maximum number of iterations
    --backend               (default cudnn)                backend
]]

print(opt)

print(c.blue '==>' ..' configuring model')
local model = dofile('models/'..opt.model..'.lua'):cuda

if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model, cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = Provider()
provider.trainData.X = provider.trainData.X:float()
provider.trainData.y = provider.trainData.y:float()
provider.testData.X = provider.testData.X:float()
provider.testData.y = provider.testData.y:float()


print('Will save at' ..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean regression accuracy (train set)', '% mean class accuracy (test set)' }
testLogger.showPlot(false)

parameters, gradParameters = model:getParameters()


print(c.blue '==>' ..' setting criterion')
criterion = nn.MSECriterion():cuda()


print(c.blue '==>' ..' configuring optimizer')
optimState = {
    learningRate = opt.learningRate
}


function train()
    model:training()
    epoch = epoch or 1

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then optimState.leanringRate = optimState.learningRate/2 end

    print(c.blue '==>'.." online epoch # " .. epoch .. " [batchSize = " .. opt.batchSize .. ']')

    local targets = torch.CudaTensor(opt.batchSize)
    local indices = torch.randperm(provider.trainData.X:size(1)):long():split(opt.batchSize)
    -- Don't need to remove last element because all the batches have equal size
    -- indices[#indices] = nil

    local tic = torch.tic()
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = provider.trainData.X:index(1, v)
        targets:copy(provider.trainData.y:index(1, v))

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f= criterion:forward(outputs, targets)    -- loss
            local df_do = criterion:backward(outputs, targets)
            model.backward(inputs, dloss_do)

            return f, gradParameters
        end

        _, fs = optim.adam(feval, parameters, optimState)

        print(fs)
    end

    epoch = epoch + 1
end


function test()
    model:evaluate()
    print(c.blue '==>'.." testing")

    local outputs = model:forward(provider.testData.X)

    print('Test loss: ', 0)

    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add(train_loss, test_loss)
        testLogger:stype('-', '-')
        testLogger:plot()
    end
end

for i = 1, opt.max_epoch do
    train()
    -- test()
end