require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class('Provider')

function Provider:__init()
    local train_size = 900
    local test_size = 100

    local img_width = 448
    local img_height = 224

    if paths.filep('dataset.t7') then
        local dataset = torch.load('dataset.t7')
        self.trainData = dataset.trainData
        self.testData = dataset.testData
    else
        self.trainData = {
            filenames = {},
            keypoints = {},
            X = torch.Tensor(train_size, 3, img_width, img_height),
            y = torch.Tensor(train_size, 8),
            size = function() return train_size end
        }

        self.testData = {
            filenames = {},
            keypoints = {},
            X = torch.Tensor(test_size, 3, img_width, img_height),
            y = torch.Tensor(test_size, 8),
            size = function() return test_size end
        }

        local labelfile = io.open(paths.concat(".", "labels.txt"), "r")
        local index = 0
        for line in labelfile:lines() do
            local filename, keypoints_str = unpack(line:split("*"))
            index = index + 1

            filename = filename:gsub("^%s*(.-)%s*$", "%1")  -- filename:strip()
            local keypoints = keypoints_str:split(" ")

            local img = image.load(paths.concat(".", "images", filename..".jpg"))
            for i = 1, 7, 2 do
                keypoints[i] = tonumber(keypoints[i]) / img:size(3)
            end
            for i = 2, 8, 2 do
                keypoints[i] = tonumber(keypoints[i]) / img:size(2)
            end
            img = image.rgb2yuv(image.scale(img, img_width, img_height))

            if index <= train_size then
                self.trainData.filenames[index] = filename..'.jpg'
                self.trainData.keypoints[index] = keypoints_str:gsub("^%s*(.-)%s*$", "%1") -- strip
                self.trainData.X[index] = img
                self.trainData.y[index] = torch.Tensor(keypoints)
            else
                self.testData.filenames[index - train_size] = filename..'.jpg'
                self.testData.keypoints[index - train_size] = keypoints_str:gsub("^%s*(.-)%s*$", "%1") -- strip
                self.testData.X[index - train_size] = img
                self.testData.y[index - train_size] = torch.Tensor(keypoints)
            end
        end
        io.close(labelfile)

        self:normalize()
        torch.save('dataset.t7', {trainData=self.trainData, testData=self.testData})
    end
end

function Provider:normalize()
    meanstdfile = io.open(paths.concat('.', 'meanstd.txt'), 'w')
    for i = 1, 3 do
        local mean = self.trainData.X:select(2,i):mean()
        local std = self.trainData.X:select(2,i):std()

        self.trainData.X:select(2,2):add(-mean)
        self.trainData.X:select(2,2):div(std)

        self.testData.X:select(2,i):add(-mean)
        self.testData.X:select(2,i):div(std)

        meanstdfile:write(mean, ' ', std, '\n')
    end
    io.close(meanstdfile)
end
