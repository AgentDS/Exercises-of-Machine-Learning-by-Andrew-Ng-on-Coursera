% ���Իع�
%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc
fprintf('================ �����Ѿƻع�4:ƽ��/�������۵÷ֵĻع� ================\n\n');
fprintf('Loading data ...\n');

%% Load Data
fn= '/Users/liangsiqi1/Documents/MATLAB/MyData/CUMCM2012A/LinearRegressionData.xlsx';
data = xlsread(fn,'LWhitewine4','B2:AC7'); % ��������
data= data(:,:)';
n= 5; % ��������
features= {'  ','����','�ܷ�','L*(D65)','a*(D65)','b*(D65)'};  % ��������
Xall = data(:, 1:n);
yall = data(:, n+1);
mall = length(yall);

mtrain= 20;
Xtrain= Xall(1:mtrain,:);
ytrain= yall(1:mtrain,:);
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[Xtrain mu sigma] = featureNormalize(Xtrain);

% Add intercept term to X
Xtrain = [ones(mtrain, 1) Xtrain];



%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(n+1, 1);
[theta, J_history] = gradientDescentMulti(Xtrain, ytrain, theta, alpha, num_iters);
% Plot the convergence graph
figure(1);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
for  k= 1:(n+1)
    disp([features{k} '    ' num2str(theta(k))]);
end
%fprintf(' %f \n', theta);
fprintf('\n');



%% ================ Part 3: Test on Test Set ================
fprintf('Test on test set ...\n');
Xtest= Xall((mtrain+1):mall,:);
ytest= yall((mtrain+1):mall,:);
mtest= mall-mtrain;
fprintf('Normalizing the features of test set ...\n');
Xtest= (Xtest-repmat(mu,mtest,1))./repmat(sigma,mtest,1);
Xtest = [ones(mtest, 1) Xtest];
predict_test= Xtest*theta;  % ��test���ݵ�yԤ��
error_test= predict_test-ytest;
abs_error= error_test./ytest;
fprintf('���Լ��ϵľ������');
abs_error
figure(2)
hold on
axis([0 mtest+1 -1 1]);
plot(([1:mtest])',abs_error,'o',[0;mtest+1],[0;0],'-b');
title('���Լ��ϵ�������ֲ�')
hold off