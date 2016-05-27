require 'torch'
require 'nn'

local Model = torch.class('Model')

function Model:__init(modelFileName)
    self.model = torch.load(paths.concat('.', 'localization', 'models', modelFileName))
    self.model:float()
    self.model:evaluate()
end

function Model:forward(input)
    self.input = input:transpose(4,3):transpose(3,2)
    return self.model:forward(self.input)
end
