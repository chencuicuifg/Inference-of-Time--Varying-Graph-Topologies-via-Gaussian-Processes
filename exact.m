function Kst = exact(hyp, x)
% EXACT Computes the Covariance Approximation Function Handle
%
% Syntax:
%   Kst = exact(hyp, x)
%
% Inputs:
%   hyp - Hyperparameters vector
%         A numeric vector containing all hyperparameters required for covariance computation.
%   x   - Input data matrix (T x N)
%         T: Number of time steps or observations
%         N: Number of features or dimensions in the data
%
% Outputs:
%   Kst - Function handle that computes ldB2_exact given a noise variance vector W
%         This handle takes W as input and returns several outputs related to the
%         log determinant of the covariance matrix and its derivatives.
%
% Description:
%   The `exact` function computes the covariance matrix `K` using the provided
%   hyperparameters and input data by calling the `covScale` function. It adds a
%   small diagonal term (1e-4) to `K` to ensure numerical stability during matrix
%   operations. The function then returns a handle `Kst`, which is a function that
%   takes a noise variance vector `W` as input and computes various quantities
%   related to the covariance matrix and its derivatives. These quantities include:
%   This modular approach allows for efficient computation of covariance-related
%   quantities necessary for probabilistic modeling and inference tasks.

    
    % Compute covariance matrix and its derivative
    [K, dK] = covScale(hyp, x);
    
    % Add small diagonal for numerical stability
    K = K + 1e-4 * eye(size(x,1));
    
    % Return function handle for ldB2_exact
    Kst = @(W) ldB2_exact(W, K, dK);
    
end

%% Sub-function: ldB2_exact
function [ldB2, solveKiW, dW, dldB2, L, triB] = ldB2_exact(W, K, dK)
% LDB2_EXACT Computes log determinant and related quantities
%
% Syntax:
%   [ldB2, solveKiW, dW, dldB2, L, triB] = ldB2_exact(W, K, dK)
%
% Inputs:
%   W  - Noise variance vector (n x 1)
%   K  - Covariance matrix (n x n)
%   dK - Function handle to compute derivative of K with respect to hyperparameters
%
% Outputs:
%   ldB2      - Log determinant of B divided by 2
%   solveKiW  - Function handle to solve B^{-1} * r
%   dW        - Derivative of ldB2 with respect to W
%   dldB2     - Function handle for derivative of ldB2
%   L         - Cholesky factor of B
%   triB      - Trace of B^{-1}
%
% Description:
%   This function computes the Cholesky factor of B = I + S * K * S,
%   where S is a diagonal matrix with sqrt(W) on the diagonal. It also
%   computes the log determinant of B, a function handle to solve systems
%   with B, the derivative of ldB2 with respect to W, and the trace of inv(B).


    % Validate W
    if ~isnumeric(W) || ~isvector(W) || any(W <= 0)
        error('Input "W" must be a positive numeric vector.');
    end
    
    % Compute number of elements
    n = numel(W);                        
    
    % Compute S = diag(sqrt(W))
    sW = sqrt(W); 
    
    % Perform Cholesky decomposition: B = L * L'
    L = chol(eye(n) + sW *sW'.* K); % Lower triangular matrix
    
    % Compute log determinant of B divided by 2
    ldB2 = sum(log(diag(L)));
    
    % Function handle to solve B^{-1} * r
    % Equivalent to inv(B) * r, computed efficiently using Cholesky factors
    solveKiW = @(r) bsxfun(@times,solve_chol(L,bsxfun(@times,r,sW)),sW);
    
    % Compute inv(B) efficiently
    invB = bsxfun(@times,1./sW,solve_chol(L,diag(sW)));
    
    % Compute derivative dW = 0.5 * diag(inv(B) * K)
    dW = sum(invB .* K, 2) / 2; % (n x 1)
    
    % Compute trace of inv(B)
    triB = trace(invB); % trace(inv(B))
    
    % Function handle for derivative of ldB2
    dldB2 = @(alpha) ldB2_deriv_exact(W, dK, invB, alpha);
end

%% Sub-function: ldB2_deriv_exact
function dhyp = ldB2_deriv_exact(W, dK, invB, alpha)
% LDB2_DERIV_EXACT Computes the derivative of ldB2 with respect to hyperparameters
%
% Syntax:
%   dhyp = ldB2_deriv_exact(W, dK, invB, alpha)
%
% Inputs:
%   W     - Noise variance vector (n x 1)
%   dK    - Function handle to compute derivative of K with respect to hyperparameters
%   invB  - Inverse of matrix B (n x n)
%   alpha - Vector involved in derivative computation (n x 1)
%
% Outputs:
%   dhyp - Derivative of ldB2 with respect to hyperparameters (vector)
%
% Description:
%   This function computes the derivative of ldB2 with respect to the hyperparameters.
%   It uses the relationship between inv(B) and the derivative of the covariance matrix dK.
%
% Example:
%   dhyp = ldB2_deriv_exact(W, dK, invB, alpha);

    % Compute R = alpha * alpha'
    R = alpha * alpha'; 
    
    % Compute the term inside dK: Q .* W - R
    % Q = invB
    term = (invB .* W) - R; % Element-wise multiplication
    
    % Compute derivative with respect to hyperparameters
    dhyp = dK(term) / 2;
end
