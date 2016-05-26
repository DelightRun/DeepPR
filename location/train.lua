require 'xlua'
require 'optim'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'provider'
require 'graph'

local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save               (default "logs")            subdirectory to save logs
    -b,--batchSize          (default 25)                batch size
    -r,--learningRate       (default 0.1)               learning rate
    -n,--nGPU               (default 2)                 number of GPUs
    --epoch_step            (default 10)                epoch step
    --model                 (default "resnet-34.t7")    model file
    --max_epoch             (default 100)                maximum number of iterations
    --savename              (default "")               model save name, nil for don't save
]]

print(opt)

print(c.blue '==>' ..' configuring model')
model = torch.load(paths.concat('.', 'models', opt.model))
cudnn.convert(model, cudnn)
if opt.nGPU > 1 then
    assert(opt.nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
    local gpus = torch.range(1, opt.nGPU):totable()
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark
    local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
    dpt.gradInput = nil
    model = dpt:cuda()
end

print(model)

print(c.blue '==>' ..' loading data')
provider = Provider()
provider.trainData.X = provider.trainData.X:cuda()
provider.trainData.y = provider.trainData.y:cuda()
provider.testData.X = provider.testData.X:cuda()
provider.testData.y = provider.testData.y:cuda()

parameters, gradParameters = model:getParameters()

inputs = torch.CudaTensor(opt.batchSize, 3, 448, 224)
targets = torch.CudaTensor(opt.batchSize, 8)
indices = torch.randperm(provider.trainData.X:size(1)):long():split(opt.batchSize)

print(c.blue '==>' ..' setting criterion')
criterion = nn.MSECriterion():cuda()


print(c.blue '==>' ..' configuring optimizer')
optimState = {
    learningRate = opt.learningRate
}

print('Will save logs at ' ..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'test average error: %', 'test maximum error: %' }
testLogger.showPlot = false

function train()
    cutorch.synchronize()

    model:training()

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then optimState.leanringRate = optimState.learningRate/2 end

    print(c.blue '==>'.." online epoch # " .. epoch .. " [batchSize = " .. opt.batchSize .. ']')

    local current_loss = 0
    local tic = torch.tic()
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)

        inputs:copy(provider.trainData.X:index(1, v))
        targets:copy(provider.trainData.y:index(1, v))

        cutorch.synchronize()
        collectgarbage()

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            model:zeroGradParameters()
            local outputs = model:forward(inputs)
            local err = criterion:forward(outputs, targets)    -- loss
            local gradOutputs = criterion:backward(outputs, targets)
            model:backward(inputs, gradOutputs)

            return err, gradParameters
        end

        _, fs = optim.adam(feval, parameters, optimState)

        if model.needsSync then
            model:syncParameters()
        end

        current_loss = current_loss + fs[1]
    end

    cutorch.synchronize()

    print(('Train loss: '..c.cyan'%.3f'..'\t time: %.2f s'):format(current_loss, torch.toc(tic)))
end

min_avg_error = 1 / 0  -- set min_avg_error to inf
best_epoch = 0
function test()
    cutorch.synchronize()
    model:evaluate()
    print(c.blue '==>'.." testing")

    local outputs = torch.Tensor(provider.testData.y:size()):cuda()
    for i = 1, provider.testData.X:size(1), opt.batchSize do
        outputs:narrow(1, i, opt.batchSize):copy(model:forward(provider.testData.X:narrow(1, i, opt.batchSize)))
        cutorch.synchronize()
    end
    cutorch.synchronize()

    local errors = provider.testData.y - outputs

    local avg_error = errors:abs():max(2):mean()
    local max_error, index = errors:abs():max(2):max(1)
    max_error = max_error[1][1]
    index = index[1][1]

    print(('Best epoch: '..c.cyan'%d'):format(best_epoch))
    print(('Minimum average error: '..c.cyan'%.3f'):format(min_avg_error))
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
        testLogger:add{avg_error}
        testLogger:style{'-'}
        testLogger:plot()
    end

    collectgarbage()

    if avg_error < min_avg_error then
        min_avg_error = avg_error
        best_epoch = epoch
        if opt.savename ~= "" then
            -- save model
            print('Save current model')
            torch.save(paths.concat('.', 'models', opt.savename), model)
        end
    end
end

epoch = 1
while epoch <= opt.max_epoch do
    train()
    test()
    epoch = epoch + 1
end

-- release memory
provider = nil
inputs = nil
targets = nil
indices = nil
model = nil
criterion = nil
collectgarbage()

if opt.savename ~= "" then
    print(c.blue '==>' ..' compressing best model')
    best_model = torch.load(paths.concat('.', 'models', opt.savename))
    best_model:clearState()
    best_model:float()
    torch.save(paths.concat('.', 'models', opt.savename), best_model)
end

print(c.blue '==>' ..' result')
print(('Best epoch: '..c.cyan'%d'):format(best_epoch))
print(('Minimum average error: '..c.cyan'%.3f'):format(min_avg_error))
