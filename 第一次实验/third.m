%%读入数据，开始训练逻辑回归线性分类器
% X为样本特征，具有两个特征，Y为样本类别
X = train_data(:, [1,2]); Y = train_data(:, 5);
Z = train_data(:, [1,2]);
% 记录样本矩阵X的大小
[m, n] = size(X);

for i = 1 : m
    tmp = X(i,1);
    X(i,1) = X(i,1) + X(i,2);
    X(i,2) = tmp * X(i,2);
end

% 样本特征中，添加一个常数特征，全部为1
X = [ones(m, 1) X];

% 创建初始化theta矩阵，大小为(n+1)*1
initial_theta = zeros(n + 1, 1);
% 使用自定义损失函数，计算得到梯度和lost
[lost, gradient] = costFunction(initial_theta, X, Y);
% 通过设置options，把fminunc的迭代次数设置为400
options = optimset('GradObj', 'off', 'MaxIter', 400);
% 使用fminunc函数，不断梯度下降，得到lost最小时的thete矩阵
[theta, lost] = fminunc(@(t)(costFunction(t, X, Y)), initial_theta, options);
 
% 打印lost和梯度矩阵
fprintf('lost: %f\n', lost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
% 画出样本和决策边界
showDecisionBoundary(theta, X, Y, Z);

% 计算得到训练样本的分类精度
train_acc = accuracy(theta, X);
fprintf('train accuracy: %f\n', mean(double(train_acc == Y)) * 100);

% 计算得到测试样本的分类精度
test_data = load("test_banknote.txt");
test_X = test_data(:, [1,2]); test_Y = test_data(:, 5);
[m, n] = size(test_X);
for i = 1 : m
    tmp = test_X(i,1);
    test_X(i,1) = test_X(i,1) + test_X(i,2);
    test_X(i,2) = tmp * test_X(i,2);
end
test_X = [ones(m, 1) test_X];
test_acc = accuracy(theta, test_X);
fprintf('test accuracy: %f\n', mean(double(test_acc == test_Y)) * 100);

%% 自定义损失函数，得到lost值和梯度值
function [lost, gradient] = costFunction(theta, X, Y)
    lambda = 1;
    L1 = 0;
    for i = 1 : size(theta,1)
        L1 = L1 + abs(theta(i,1));
    end
    L1 = lambda * L1;
    
    n = length(Y); % 训练样本的数目
    lost = 0;
    gradient = zeros(size(theta));
    sig = sigmoid(X * theta);
    lost = (1/n * (-Y' * log(sig) - (1 - Y') * log(1 - sig))) + L1; % 计算lost
    gradient = 1/n * (X' * (sig - Y)); % 计算梯度
end

%% sigmoid函数
function ans = sigmoid(X)
    ans = zeros(size(X));
    ans = 1 ./ (1 + exp(-X));
end

%% 按类别展示样本数据
function showData(X, Y)    
    figure;
    hold on;
    class0 = find(Y == 0);
    class1 = find(Y == 1);
    plot(X(class0,1), X(class0,2), 'ro');
    plot(X(class1,1), X(class1,2), 'go');
    axis([-8, 8, -15, 15])
    xlabel('LR linear classifier')
    hold off;
end

%% 展示决策边界
function showDecisionBoundary(theta, X, Y, Z)
    % 展示类别数据
    showData(Z, Y);
    hold on
    % 计算得到决策边界的两个点，画出一条线
    plot_x = [min(X(:,2)) - 2,  max(X(:,2)) + 2];
    plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));
    plot(plot_x, plot_y)
    hold off
end

%% 根据训练得到的theta矩阵，把样本计算一次，得到结果，用于比较
function acc = accuracy(theta, X)
    m = size(X, 1); % Number of training examples
    acc = zeros(m, 1);
    acc = round(sigmoid(X * theta));
end