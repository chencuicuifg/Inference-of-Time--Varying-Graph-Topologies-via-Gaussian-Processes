function [maxMPoutput] = multipleIni_SSM_FBGPs(x, y, L, varargin)
% multipleIni_SSM_fBGPs Runs SSM_FBGPs Multiple Times with Different Initializations
%
% Syntax:
%   maxMPoutput = multipleIni_SSM_FBGPs(x, y, L, 'ParamName', ParamValue, ...)
%
% Inputs:
%   x, y, L   - Same as in SSM_FBGPs
%   varargin  - Optional name-value pair arguments for SSM_FBGPs
%
% Outputs:
%   maxMPoutput - The output structure from SSM_FBGPs with the highest MP value

    % Number of initializations to run
    numIniPoints = 5;

    % Preallocate cell array for outputs
    output = cell(numIniPoints, 1);

    % Preallocate MP array
    MP = zeros(numIniPoints, 1);

    % Parallel loop for multiple initializations
    parfor i = 1:numIniPoints
        % Call SSM_FBGPs with expanded varargin
        output{i} = SSM_FBGPs(x, y, L, varargin{:});
        
        % Store the MP value
        MP(i) = output{i}.MP;
    end 

    % Find the index with the maximum MP
    [~, maxMPInd] = max(MP);

    % Retrieve the corresponding output
    maxMPoutput = output{maxMPInd};
end
