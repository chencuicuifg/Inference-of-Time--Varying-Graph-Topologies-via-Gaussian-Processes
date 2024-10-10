function [] = Visualization(output)

% --- Visualization of Mean Weights with Shaded Uncertainty ---

% Extract mean and standard deviation matrices
mean_w = output.sampleMean.w;  % Size: [L x N]
std_w = output.sampleStd.w;    % Size: [L x N]

% Determine number of windows and nodes
[L, N] = size(mean_w);  % [L x N]

% Define colors for each node for better distinction
% Use a colormap with N distinct colors
colors = lines(N);  % 'lines' colormap provides distinct colors

% Define window indices
windows = 1:L;

% Create a figure for overlayed plots with shaded uncertainty
figure;
hold on;
grid on;

% Loop through each node to plot mean and shaded uncertainty
for node = 1:N
    % Extract mean and std for the current node
    node_mean = mean_w(:, node)';  % Transpose to make it a row vector [1 x L]
    node_std = std_w(:, node)';    % Transpose to make it a row vector [1 x L]
    
    % Compute upper and lower bounds
    upper = node_mean + node_std;  % [1 x L]
    lower = node_mean - node_std;  % [1 x L]
    
    % Check if upper and lower are the same length as windows
    if length(upper) ~= length(windows) || length(lower) ~= length(windows)
        error('Length of upper and lower bounds must match the number of windows.');
    end
    
    % Create shaded area between (mean + std) and (mean - std)
    % 'fliplr' is used to create a closed loop for the fill
    fill([windows, fliplr(windows)], [upper, fliplr(lower)], colors(node,:), ...
         'FaceAlpha', 0.2, 'EdgeColor', 'none','DisplayName', sprintf('Uncertainty of node %d', node));
    
    % Plot the mean weight line
    plot(windows, node_mean, 'Color', colors(node,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Node %d', node));
end

% Customize plot
xlabel('Window');
ylabel('Mean Weight');
title('Mean Weights Across Windows with One Standard Deviation');
legend('Location', 'best');
hold off;

% --- End of Visualization ---
end