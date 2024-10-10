function output = SSM_FBGPs(x, y, L, varargin)
    % SSM_FBGPs Perform Stochastic Sampling for State-Space Model - FBGPs Model
    %
    % Syntax:
    %   output = SSM_FBGPs(x, y, L)
    %   output = SSM_FBGPs(x, y, L, 'ParameterName', ParameterValue, ...)
    %
    % Inputs:
    %   x   - Training data matrix of size (T x N)
    %         T: Number of time steps or observations
    %         N: Dimension of the data (number of features)
    %   y   - Target data vector of size (T x 1)
    %   L   - Number of windows (segments) to divide the data into
    %         Typically, L = T / window_size
    %
    % Name-Value Pair Arguments:
    %   'psv'        - Prior of the signal variance (default: 'halfnormal')
    %                  Options: 'constant', 'halfnormal', 'inversegamma'
    %   'pnv'        - Prior of the noise variance (default: 'halfnormal')
    %                  Options: 'constant', 'halfnormal', 'inversegamma'
    %   'BI'         - Burn-in sample number (default: 5000)
    %   'numSamples' - Number of samples to draw (default: 5000)
    %   'sigmal'     - Gaussian random walk standard deviation (default: 0.1)
    %   'lambda'     - Weight of the Gaussian random walk prior (default: 1)
    %   'eta'        - Weight of the sparse prior on the first window (default: 1)
    %   'ini'        - Initial value of the parameters, size (D x 1)
    %                  log([w; sigma_n; sigma_v])
    %                  Default: log([normrnd(2,0.5,1,1)*ones(L*N,1); std(y); std(y)])
    %
    % Outputs:
    %   output - Structure containing the following fields:
    %     .sampleMean - Mean of the sampled parameters
    %     .sampleStd  - Standard deviation of the sampled parameters
    %     .MP         - Marginal Posterior Evaluation
    %     .samples    - All drawn samples
    %     .infor      - Additional information from the sampling process
    %
    % Description:
    %   The `SSM_FBGPs` function performs stochastic sampling for the Finite Bayesian Gaussian Processes (FBGPs) model.
    %   It leverages Hamiltonian Monte Carlo (HMC) sampling to draw samples from the posterior distribution of the model's parameters.
    %   The function supports various priors for signal and noise variances and allows customization through multiple parameters.
    %
    %   **Key Steps:**
    %     1. **Input Parsing and Validation:**
    %        - Utilizes an input parser to handle required inputs and optional name-value pair arguments.
    %        - Sets default values for optional parameters and validates their types and ranges.
    %
    %     2. **Prior Definitions:**
    %        - Defines prior probability density functions (PDFs) and their derivatives for both signal and noise variances based on user-specified types.
    %
    %     3. **Inference Setup:**
    %        - Constructs a marginal posterior function handle `INMP` using the `MPinf` function, which incorporates priors and likelihood.
    %        - Initializes an HMC sampler with the specified step size and gradient checking.
    %
    %     4. **Sampling Process:**
    %        - Draws samples using the HMC sampler, accounting for burn-in periods and the desired number of samples.
    %        - Computes the marginal posterior `MP` based on the mean of the sampled parameters.
    %
    %   This function is essential for probabilistic modeling and Bayesian inference in Gaussian Process frameworks, providing robust sampling mechanisms to explore the posterior distribution of model parameters.
    %
    % Examples:
    %   % Perform stochastic sampling with default parameters
    %   output = SSM_FBGPs(x, y, L);
    %
    %   % Perform stochastic sampling with custom parameters
    %   output = SSM_FBGPs(x, y, L, ...
    %       'psv', 'inversegamma', ...
    %       'pnv', 'halfnormal', ...
    %       'BI', 10000, ...
    %       'numSamples', 20000, ...
    %       'sigmal', 0.2, ...
    %       'lambda', 1.5, ...
    %       'eta', 0.8, ...
    %       'ini', log([ones(L*size(x, 2), 1) * 1.5; 0.5; 0.5]));
    %
    % See Also:
    %   MPinf, hmcSampler, drawSamples, getPrior
    
%% Scale Inputs to [-1, 1] Using Common Scalars
%     % Compute the global minimum and maximum across x and y
%     global_min = min([min(x(:)), min(y(:))]);
%     global_max = max([max(x(:)), max(y(:))]);
% 
%     % Prevent division by zero in case global_max equals global_min
%     if global_max == global_min
%         error('Cannot scale data because global_max equals global_min.');
%     end
% 
%     % Scale x and y to [-1, 1] using the same global_min and global_max
%     x = 2 * (x - global_min) / (global_max - global_min) - 1;
%     y = 2 * (y - global_min) / (global_max - global_min) - 1;
    %% Parse and Validate Inputs
    params = parseInputs(x, y, L, varargin{:});
    
    %% Define Priors
    [psvpdf, dpsvpdf] = getPrior(params.psv, 'signal');
    [pnvpdf, dpnvpdf] = getPrior(params.pnv, 'noise');
    
    %% Inference Setup
    % Define the marginal posterior function handle
    INMP = @(hyp) MPinf(hyp, x, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, ...
                        params.L, params.sigmal, params.lambda, params.eta);
    
    % Initialize HMC sampler with the defined marginal posterior
    smpMP = hmcSampler(INMP, params.ini, ...
                       "CheckGradient", true, 'StepSize', 0.03);
    
    %% Sampling Process
    % Draw samples using the HMC sampler
    Samples = drawSamples(smpMP, 'NumSamples', params.numSamples, ...
                                'burnin', params.BI);
    
    %% Compute Statistics
    [sampleMean, sampleStd] = computeStatistics(Samples, L, size(x, 2));
    
    %% Compute Marginal Posterior
    MP = INMP(mean(Samples)');
    
    %% Prepare Output Structure
    output.sampleMean = sampleMean;
    output.sampleStd  = sampleStd;
    output.MP         = MP;
    output.samples    = Samples;
    
end

%% Sub-function: Parse Inputs
function params = parseInputs(x, y, L, varargin)
    % PARSEINPUTS Parses and Validates Inputs for SSM_FBGPs Function
    %
    % Syntax:
    %   params = parseInputs(x, y, L, 'ParamName', ParamValue, ...)
    %
    % Inputs:
    %   x       - Training data matrix (T x N)
    %   y       - Target data vector (T x 1)
    %   L       - Number of windows (scalar)
    %   varargin - Optional name-value pair arguments
    %
    % Outputs:
    %   params  - Struct containing all parsed parameters with default values applied
    %
    % Description:
    %   This function utilizes MATLAB's inputParser to handle required inputs and
    %   optional name-value pair arguments. It sets default values for optional
    %   parameters and validates the inputs to ensure they meet expected criteria.
    %
    %   If the 'ini' parameter is not provided, it initializes the parameters based
    %   on the provided data. If 'ini' is provided, it validates the size to ensure
    %   compatibility with the model.

    %% Initialize Input Parser
    p = inputParser;
    p.CaseSensitive = false;
    p.FunctionName = 'SSM_FBGPs';
    
    % Required arguments
    validateData = @(z) isnumeric(z) && ismatrix(z);
    validateTarget = @(z) isnumeric(z) && isvector(z);
    validateL = @(z) isnumeric(z) && isscalar(z) && z > 0;
    
    addRequired(p, 'x', validateData);
    addRequired(p, 'y', validateTarget);
    addRequired(p, 'L', validateL);
    
    % Optional Name-Value pairs
    validPriors = {'constant', 'halfnormal', 'inversegamma'};
    checkPsv = @(z) any(validatestring(z, validPriors));
    checkPnv = @(z) any(validatestring(z, validPriors));
    
    addParameter(p, 'psv', 'halfnormal', checkPsv);
    addParameter(p, 'pnv', 'halfnormal', checkPnv);
    addParameter(p, 'BI', 5000, @(z) isnumeric(z) && isscalar(z) && z >= 0);
    addParameter(p, 'numSamples', 5000, @(z) isnumeric(z) && isscalar(z) && z > 0);
    addParameter(p, 'sigmal', 0.1, @(z) isnumeric(z) && isscalar(z) && z > 0);
    addParameter(p, 'lambda', 1, @(z) isnumeric(z) && isscalar(z));
    addParameter(p, 'eta', 1, @(z) isnumeric(z) && isscalar(z));
    addParameter(p, 'ini', [], @(z) isempty(z) || (isnumeric(z) && isvector(z)));
    
    % Parse inputs
    parse(p, x, y, L, varargin{:});
    params = p.Results;
    
    % Set default 'ini' if not provided
    if isempty(params.ini)
        N = size(x, 2);
        wini = abs(normrnd(2, 0.5, 1, 1)); % Ensure positivity
        params.ini = log([repmat(wini, L * N, 1); ...
                         std(y); std(y)]);
    else
        % Validate 'ini' size
        expectedSize = (size(x, 2) * L + 2) * 1;
        if numel(params.ini) ~= expectedSize
            error('Initial parameter "ini" must be of size (%d x 1).', expectedSize);
        end
    end
end

%% Sub-function: Define Priors
function [pdf, dpdf] = getPrior(priorType, varianceType)
    % GETPRIOR Defines the Prior PDF and Its Derivative Based on the Prior Type
    %
    % Syntax:
    %   [pdf, dpdf] = getPrior(priorType, varianceType)
    %
    % Inputs:
    %   priorType    - Type of the prior ('constant', 'halfnormal', 'inversegamma')
    %   varianceType - Type of variance ('signal', 'noise') for 'inversegamma' prior
    %
    % Outputs:
    %   pdf  - Function handle for the prior probability density function
    %   dpdf - Function handle for the derivative of the prior PDF
    %
    % Description:
    %   This function returns the appropriate prior PDF and its derivative based on
    %   the specified `priorType` and `varianceType`. It supports 'constant',
    %   'halfnormal', and 'inversegamma' priors.
    %
    %   For 'inversegamma' priors, different parameters can be set based on whether
    %   the variance is for 'signal' or 'noise'.

    switch lower(priorType)
        case 'constant'
            % Constant prior: flat distribution
            pdf = @(v) 0;
            dpdf = @(v) 0;
            
        case 'halfnormal'
            % Half-Normal prior
            % Assuming the scale parameter is embedded within the function
            % Typically, half-normal has a scale parameter sigma
            sigma = 1; % You can make sigma a parameter if needed
            pdf = @(v) log(2) - log(sigma * sqrt(2 * pi)) - (v.^2) / (2 * sigma^2);
            dpdf = @(v) (-v) / (sigma^2);
            
        case 'inversegamma'
            % Inverse Gamma prior
            % Parameters can be adjusted based on variance type
            switch lower(varianceType)
                case 'signal'
                    a = 2;    % Shape parameter
                    b = 1;    % Scale parameter
                case 'noise'
                    a = 3;    % Shape parameter
                    b = 1;    % Scale parameter
                otherwise
                    error('Unknown variance type: %s', varianceType);
            end
            pdf = @(v) a * log(b) - gammaln(a) - (a + 1) .* log(v) - b ./ v;
            dpdf = @(v) -(a + 1) ./ v + b ./ v.^2;
            
        otherwise
            error('Unsupported prior type: %s', priorType);
    end
end

%% Sub-function: Compute Statistics
function [sampleMean, sampleStd] = computeStatistics(Samples, L, N)
    % COMPUTESTATISTICS Computes the mean and standard deviation of the samples
    %
    % Syntax:
    %   [sampleMean, sampleStd] = computeStatistics(Samples, L, N)
    %
    % Inputs:
    %   Samples - Matrix of sampled parameters (numSamples x (LN+2))
    %   L       - Number of windows
    %   N       - Number of nodes
    %
    % Outputs:
    %   sampleMean - Mean of the sampled parameters
    %   sampleStd  - Standard deviation of the sampled parameters

    
    wSamples = reorderColumnsByNode(Samples,N);
    sigma_n_samples = Samples(:, L*N + 1);
    sigma_v_samples = Samples(:, L*N + 2);
    
    % Compute mean and std for w across samples
    sampleMean.w = reshape(mean(exp(wSamples), 1),L,N); % (L x N)
    
    sampleStd.w = reshape(std(exp(wSamples), 0, 1),L,N); % (1 x L x N)
    
    % Compute mean and std for noise and signal variances
    sampleMean.sigma_n = mean(exp(sigma_n_samples));
    sampleMean.sigma_v = mean(exp(sigma_v_samples));
    
    sampleStd.sigma_n = std(exp(sigma_n_samples));
    sampleStd.sigma_v = std(exp(sigma_v_samples));
    
    % Optionally, you can flatten or structure these statistics as needed
end

function wSamples = reorderColumnsByNode(Samples, N)
    % reorderColumnsByCategory Reorganizes matrix columns by nodes.
    %
    % Syntax:
    %   M_new = reorderColumnsByCategory(Samples, N)
    %
    % Inputs:
    %   Samples - Original matrix with columns in repeating group order.
    %   N - Number of nodes.
    %
    % Outputs:
    %   wSamples - Matrix with columns grouped by nodes.

    % Get the total number of columns
    [~, D] = size(Samples);
    D = D-2; % not consider the last two column, signal variance and noise variance
    % Calculate the number of groups
    if mod(D, N) ~= 0
        error('Number of columns (%d) is not divisible by the number of nodes N (%d).', D, N);
    end
    G = D / N;
    
    % Create a matrix of column indices reshaped into G rows and N columns
    reshapedIndices = reshape(1:D, N, G)';  % Size: [G x N]
    
    % Linearize the reshaped indices column-wise to group by category
    newOrder = reshapedIndices(:)';  % Row vector
    
    % Reorder the columns of wSamples
    wSamples = Samples(:, newOrder);
end

