require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class('Provider')

function Provider:__init()
    local train_size = 800
    local test_size = 200

    if paths.filep('dataset.t7') then
        local dataset = torch.load('dataset.t7')
        self.trainData = dataset.trainData
        self.testData = dataset.testData
    else
        self.trainData = {
            filenames = {},
            keypoints = {},
            X = torch.Tensor(800, 3, 224, 112),
            y = torch.Tensor(800, 8),
            size = function() return train_size end
        }

        self.testData = {
            filenames = {},
            keypoints = {},
            X = torch.Tensor(200, 3, 224, 112),
            y = torch.Tensor(200, 8),
            size = function() return test_size end
        }

        local labelfile = io.open(paths.concat("..", "labels.txt"), "r")
        local index = 0
        for line in labelfile:lines() do
            local filename, keypoints_str = unpack(line:split("*"))
            index = index + 1

            filename = filename:gsub("^%s*(.-)%s*$", "%1")  -- filename:strip()
            local keypoints = keypoints_str:split(" ")

            local img = image.load(paths.concat("..", "data", filename..".jpg"))
            for i = 1, 7, 2 do
                keypoints[i] = tonumber(keypoints[i]) / img:size(3)
            end
            for i = 2, 8, 2 do
                keypoints[i] = tonumber(keypoints[i]) / img:size(2)
            end
            img = image.rgb2yuv(image.scale(img, 224, 112))

            if index <= train_size then
                self.trainData.filenames[index] = filename..'.jpg'
                self.trainData.keypoints[index] = keypoints_str:gsub("^%s*(.-)%s*$", "%1")
                self.trainData.X[index] = img
                self.trainData.y[index] = torch.Tensor(keypoints)
            else
                self.testData.filenames[index - train_size] = filename..'.jpg'
                self.testData.keypoints[index - train_size] = keypoints_str:gsub("^%s*(.-)%s*$", "%1")
                self.testData.X[index - train_size] = img
                self.testData.y[index - train_size] = torch.Tensor(keypoints)
            end
        end
        io.close(labelfile)
        torch.save('dataset.t7', {trainData=self.trainData, testData=self.testData})
    end
end

function Provider:detectEdge()
    local sobel_vertical = torch.Tensor({{-1, 0, -1}, {-2, 0, -2}, {-1, 0, -1}})
    local sobel_horizontal = torch.Tensor({{-1, -2, -1}, {0, 0, 0}, {-1, -2, -1}})

    -- preprocess trainSet
    for i = 1, self.trainData:size() do
        if i % 50 == 0 then
            xlua.progress(i, self.trainData:size())
        end

        local yuv = self.trainData.X[i]
        local v = image.convolve(yuv[1], sobel_vertical, 'same')
        local h = image.convolve(yuv[1], sobel_horizontal, 'same')
        yuv[1] = (v:pow(2) + h:pow(2)):sqrt()

        self.trainData.X[i] = yuv
    end
    -- preprocess testSet
    for i = 1, self.testData:size() do
        if i % 50 == 0 then
            xlua.progress(i, self.testData:size())
        end

        local yuv = self.testData.X[i]
        local v = image.convolve(yuv[1], sobel_vertical, 'same')
        local h = image.convolve(yuv[1], sobel_horizontal, 'same')
        yuv[1] = (v:pow(2) + h:pow(2)):sqrt()

        self.testData.X[i] = yuv
    end
end

function Provider:normalize()
    local normalization = nn.SpatialSubtractiveNormalization(1, image.gaussian1D(7))

    -- preprocess trainSet
    for i = 1, self.trainData:size() do
        if i % 50 == 0 then
            xlua.progress(i, self.trainData:size())
        end

        -- rgb -> yuv
        local yuv = self.trainData.X[i]
        -- normailize y locally
        yuv[1] = normalization(yuv[{{1}}])
        self.trainData.X[i] = yuv
    end
    -- normalize u globally:
    local mean_u = self.trainData.X:select(2,2):mean()
    local std_u = self.trainData.X:select(2,2):std()
    self.trainData.X:select(2,2):add(-mean_u)
    self.trainData.X:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = self.trainData.X:select(2,3):mean()
    local std_v = self.trainData.X:select(2,3):std()
    self.trainData.X:select(2,3):add(-mean_v)
    self.trainData.X:select(2,3):div(std_v)

    self.trainData.mean_u = mean_u
    self.trainData.std_u = std_u
    self.trainData.mean_v = mean_v
    self.trainData.std_v = std_v

    -- preprocess testSet
    for i = 1, self.testData:size() do
        if i % 50 == 0 then
            xlua.progress(i, self.testData:size())
        end

        -- rgb -> yuv
        local rgb = self.testData.X[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally
        yuv[1] = normalization(yuv[{{1}}])
        self.testData.X[i] = yuv
    end
    --normalize u globally
    self.testData.X:select(2,2):add(-mean_u)
    self.testData.X:select(2,2):div(std_u)
    --normalize v globally
    self.testData.X:select(2,3):add(-mean_v)
    self.testData.X:select(2,3):div(std_v)
end
