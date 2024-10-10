function [X2,y2,wall] = example1Data()
    % example1Data Generates synthetic dataset for Example 1.
    %
    % [X2, y2] = example1Data() generates a dataset with N samples divided
    % into L blocks. It returns the augmented feature matrix X2 and the response
    % vector y2.
    %
    % Inputs:
    %   T - Total number of samples (must be divisible by L)
    %   L - Number of blocks
    %
    % Outputs:
    %   X2 - (T+1) x 5 augmented feature matrix (original features + y2)
    %   y  - (T+1) x 1 response vector

T =400;
L = 8;
ws = T/L;
K = zeros(T,T);
sigmav = 0.1;

w1 = [1.3,1.2,1.1,1,0.9,0.8,0.7,0.6];
w2 = [.01,.01,.01,.1,0.2,0.3,0.3,0.4];
w3 = [.6,.5,.4,.3,.2,.1,0,0];
w4 = .6*sin(linspace(0,2,L))+0.3;
wall = [w1;w2;w3;w4]';

z =  normrnd(0,1,T+1,1);
q = normrnd(0,1,T+1,1);
s = normrnd(0,1,T+1,1);
v = normrnd(0,1,T+1,1);
X = [z,q,s,v]; x = X(1:end-1,:);

for i = 1:L
    [K((i-1)*ws+1:i*ws,(i-1)*ws+1:i*ws),~,~]= covSE(log(wall(i,:)'),x((i-1)*ws+1:i*ws,:));
end 
y =  mvnrnd(zeros(T,1),K)'+normrnd(0,sigmav,T,1);
y2 = [y(1);y ];
X2 = [X,y2];
end