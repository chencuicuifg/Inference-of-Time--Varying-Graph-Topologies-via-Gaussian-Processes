function [AlllZ, AlldlZ] = MPinf(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, L, sigmal, lambda, eta)
% MPinf Computes the Log Posterior and Its Derivatives
%
% Syntax:
%   [AlllZ, AlldlZ] = MPinf(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, L, sigmal, lambda, eta)
%
% Inputs:
%   hyp      - Hyperparameters vector [ll_1; ...; ll_k; sv; nv]
%   x        - Training data matrix of size (T x N)
%              T: Data length
%              N: Dimension of the data
%   y        - Target data vector of size (T x 1)
%   psvpdf   - Function handle for the prior PDF of signal variance
%   dpsvpdf  - Function handle for the derivative of the prior PDF of signal variance
%   pnvpdf   - Function handle for the prior PDF of noise variance
%   dpnvpdf  - Function handle for the derivative of the prior PDF of noise variance
%   L        - Number of windows (segments)
%   sigmal   - Standard deviation of the Gaussian random walk
%   lambda   - Weight of the Gaussian random walk prior
%   eta      - Weight of the sparse prior on the first window
%
% Outputs:
%   AlllZ   - Total log posterior
%   AlldlZ  - Derivatives of the total log posterior
%
% Description:
%   This function computes the log posterior (AlllZ) and its
%   derivatives (AlldlZ) for a given set of hyperparameters. It incorporates
%   prior probabilities and the likelihood of the data segmented into windows.
%
% Example:
%   [AlllZ, AlldlZ] = MPinf(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, L, sigmal, lambda, eta);

    %% Input Validation
    validateInputs(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, L, sigmal, lambda, eta);

    %% Initialize Variables
    [N, D] = size(x); % N: Time steps, D: Data dimensions
    ws = ceil(N / L); % Window size
    sv = hyp(end-1);  % Signal variance
    nv = hyp(end);    % Noise variance

    % Compute noise variance for Gaussian likelihood
    sn2 = exp(2 * hyp(end));
    W = ones(ws, 1) / sn2; % Noise variance matrix

    % Extract and reshape weights
    lw = hyp(1:end-2); % Log-weights
    elw = exp(lw);     % Exponentiated weights
    elwmatrix = reshape(elw, D, L); % Reshape to (D x L)

    %% Compute Prior Probability and Its Derivative
    [ppdf, dppdf] = computePrior(elwmatrix, sv, nv, lambda, eta, sigmal, psvpdf, dpsvpdf, pnvpdf, dpnvpdf);

    %% Compute Likelihood and Its Derivatives
    [lZ, dlZ_cov, dlZ_n] = computeLikelihood(hyp, x, y, elwmatrix, W, L, ws, sn2);

    %% Aggregate Log Posterior and Its Derivatives
    AlllZ = sum(lZ) + ppdf;
    AlldlZ = [reshape(dlZ_cov(1:end-1, :), [], 1); ...
               sum(dlZ_cov(end, :)); ...
               sum(dlZ_n)] + dppdf;
end

%% Sub-function: Validate Inputs
function validateInputs(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, L, sigmal, lambda, eta)
    % Validate hyperparameters
    validateattributes(hyp, {'numeric'}, {'vector'}, mfilename, 'hyp', 1);

    % Validate x and y
    validateattributes(x, {'numeric'}, {'2d'}, mfilename, 'x', 2);
    validateattributes(y, {'numeric'}, {'vector'}, mfilename, 'y', 3);
    if length(y) ~= size(x, 1)
        error('Length of y must match the number of rows in x.');
    end

    % Validate function handles for priors
    validateattributes(psvpdf, {'function_handle'}, {}, mfilename, 'psvpdf', 4);
    validateattributes(dpsvpdf, {'function_handle'}, {}, mfilename, 'dpsvpdf', 5);
    validateattributes(pnvpdf, {'function_handle'}, {}, mfilename, 'pnvpdf', 6);
    validateattributes(dpnvpdf, {'function_handle'}, {}, mfilename, 'dpnvpdf', 7);

    % Validate scalar inputs
    validateattributes(L, {'numeric'}, {'scalar', 'positive', 'integer'}, mfilename, 'L', 8);
    validateattributes(sigmal, {'numeric'}, {'scalar', 'positive'}, mfilename, 'sigmal', 9);
    validateattributes(lambda, {'numeric'}, {'scalar'}, mfilename, 'lambda', 10);
    validateattributes(eta, {'numeric'}, {'scalar'}, mfilename, 'eta', 11);
end

%% Sub-function: Compute Prior Probability and Its Derivative
function [ppdf, dppdf] = computePrior(elwmatrix, sv, nv, lambda, eta, sigmal, psvpdf, dpsvpdf, pnvpdf, dpnvpdf)
    D = size(elwmatrix, 1);
    L = size(elwmatrix, 2);
    
    % Constant term in prior probability
    constant = -lambda * (L - 1) * D / 2 * log(2 * pi * sigmal^2);
    
    % Log-prior probability
    lpdf = sum(-[eta * elwmatrix(:, 1), ...
                lambda * 0.5 .* (elwmatrix(:, 2:end) - elwmatrix(:, 1:end-1)).^2 / sigmal^2], ...
                'all');
    plpdf = constant + lpdf;
    
    % Total prior PDF including signal and noise variances
    ppdf = plpdf + psvpdf(sv) + pnvpdf(nv);
    
    % Derivative of the prior PDF
    % Partial derivatives with respect to elwmatrix
    dplpdf2 =lambda.*...
        ([-eta/lambda-elwmatrix(:,1)./sigmal.^2, [-2.*elwmatrix(:,2:end-1),-1.*elwmatrix(:,end)]./sigmal.^2]...
        + [elwmatrix(:,2:end),zeros(D,1)]./sigmal.^2 ...
        + [zeros(D,1),elwmatrix(:,1:end-1)]./sigmal.^2).*elwmatrix;    
    dppdf = [reshape(dplpdf2, [], 1); dpsvpdf(sv); dpnvpdf(nv)];
end

%% Sub-function: Compute Likelihood and Its Derivatives
function [lZ, dlZ_cov, dlZ_n] = computeLikelihood(hyp, x, y, elwmatrix, W, L, ws, sn2)
    % Initialize outputs
    lZ = zeros(L, 1);
    D = size(elwmatrix, 1);
    dlZ_cov = zeros(D + 1, L); % Assuming dlZ_cov has D+1 rows
    dlZ_n = zeros(L, 1);
    
    for liter = 1:L
        % Extract hyperparameters for the current window
        lw_segment = hyp((liter-1)*D + 1 : liter*D);
        sv_segment = hyp(end-1);
        
        % Setup covariance approximation
        Kst = exact([lw_segment; sv_segment], x((liter-1)*ws + 1 : liter*ws, :));
        
        % Obtain functionalities based on W
        [ldB2, solveKiW, dW, dhyp] = Kst(W);
        
        % Compute alpha for the current window
        alpha = solveKiW(y((liter-1)*ws + 1 : liter*ws));
        
        % Compute log marginal likelihood for the current window
        lZ(liter) = -(y((liter-1)*ws + 1 : liter*ws)' * alpha / 2 + ...
                      ldB2 + ws * log(2 * pi * sn2) / 2);
        
        % Compute derivatives with respect to hyperparameters
        dlZ_cov(:, liter) = -dhyp(alpha);
        
        % Compute derivatives with respect to noise variance
        dlZ_n(liter) = sn2 * (alpha' * alpha) + 2 * sum(dW) / sn2 - ws;
    end
end
