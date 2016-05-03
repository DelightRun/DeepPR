require 'xlua'
require 'optim'
require 'cunn'
require 'cudnn'
require 'provider'

local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save               (default "logs")            subdirectory to save logs
    -b,--batchSize          (default 25)                batch size
    -r,--learningRate       (default 0.001)             learning rate
    -n,--nGPU               (default 1)                 number of GPUs
    --epoch_step            (default 10)                epoch step
    --depth                 (default 152)               model depth
    --max_epoch             (default 30)                maximum number of iterations
]]

print(opt)

print(c.blue '==>' ..' configuring model')
local model = torch.load(paths.concat('.', 'models', 'resnet-'..opt.depth..'.t7'))
if opt.nGPU > 1 then
    assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
    local model_single = model
    model = nn.DataParallelTable(1)
    for i=1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(model_single:clone():cuda(), i)
    end
end
cudnn.convert(model, cudnn)

print(model)

print(c.blue '==>' ..' loading data')
provider = Provider()
provider:normalize()
provider.trainData.X = provider.trainData.X:cuda()
provider.trainData.y = provider.trainData.y:cuda()
provider.testData.X = provider.testData.X:cuda()
provider.testData.y = provider.testData.y:cuda()

print('Will save at ' ..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'test average error: %', 'test maximum error: %' }
testLogger.showPlot = false

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

    local targets = torch.CudaTensor(opt.batchSize, 8)
    local indices = torch.randperm(provider.trainData.X:size(1)):long():split(opt.batchSize)
    -- Don't need to remove last element because all the batches have equal size
    -- indices[#indices] = nil

    local current_loss = 0
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
            model:backward(inputs, df_do)

            return f, gradParameters
        end

        _, fs = optim.adam(feval, parameters, optimState)

        current_loss = current_loss + fs[1]
    end

    print(('Train loss: '..c.cyan'%.3f'..'\t time: %.2f s'):format(current_loss, torch.toc(tic)))

    epoch = epoch + 1
end


function test()
    model:evaluate()
    print(c.blue '==>'.." testing")

    local outputs = torch.Tensor(provider.testData.y:size()):cuda()
    for i = 1, provider.testData.X:size(1), opt.batchSize do
        outputs:narrow(1, i, opt.batchSize):copy(model:forward(provider.testData.X:narrow(1, i, opt.batchSize)))
    end
    local error = provider.testData.y - outputs

    local avg_error = error:abs():max(2):mean()
    local max_error, index = error:abs():max(2):max(1)
    max_error = max_error[1][1]
    index = index[1][1]

    print(('Test average error: '..c.cyan'%.3f'):format(avg_error))
    print(('Test maximum error: '..c.cyan'%.3f'):format(max_error))

    print(('Index: '..c.cyan'%d'):format(index))

    print(('filename: '..c.cyan'%s'):format(provider.testData.filenames[index]))
    print(('keypoints: '..c.cyan'%s'):format(provider.testData.keypoints[index]))
    print('Expected output: ')
    print(provider.testData.y[index])
    print('Actual output: ')
    print(outputs[index])

    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add{avg_error, max_error}
        testLogger:style{'-', '-'}
        testLogger:plot()
    end
end

for i = 1, opt.max_epoch do
    train()
    test()
end
