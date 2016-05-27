require 'torch'
require 'nn'

local Model = torch.class('Model')

function Model:__init(modelFileName, CPU_MODE)
    self.model = torch.load(paths.concat('.', 'models', modelFileName))
    if not CPU_MODE then
        require 'cunn'
        require 'cudnn'
        self.model:cuda()
        cudnn.convert(self.model, cudnn)
        cudnn.fastest = true
    end
end

function Model:forward(input)
    self.input = input:transpose(4,3):transpose(3,2)
    return self.model:forward(self.input)
end
