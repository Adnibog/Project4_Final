function [P, s, X_new, per] = PCA_C(data, type)
    % PCA function for dimensionality reduction
    % Inputs:
    % - data: Feature matrix (rows = features, columns = samples)
    % - type: 1 for eigen decomposition, 2 for SVD
    %
    % Outputs:
    % - P: Principal component matrix (eigenvectors)
    % - s: Eigenvalues or singular values
    % - X_new: Projected data
    % - per: Explained variance percentage for each component
    
    % Remove mean from data
    [feature_n, sample_n] = size(data);
    data = data - mean(data, 2) * ones(1, sample_n); % Centering data
    
    if type == 2
        % PCA using Singular Value Decomposition (SVD)
        [U, S, ~] = svd(data, 'econ'); % SVD decomposition
        s = diag(S).^2 / (sample_n - 1); % Eigenvalues from singular values
        tv = sum(s); % Total variance
        
        per = s / tv; % Explained variance proportion
        tper = cumsum(per); % Cumulative explained variance
        
        P = U; % Principal components (eigenvectors)
        figure;
        plot(tper, 'o-', 'LineWidth', 1.5);
        title('Cumulative Explained Variance (SVD)');
        xlabel('Number of Components');
        ylabel('Variance Explained');
        grid on;

    elseif type == 1
        % PCA using Eigen decomposition
        cov_data = data * data' / (sample_n - 1); % Covariance matrix
        [U, D] = eig(cov_data); % Eigen decomposition
        s = diag(D); % Extract eigenvalues
        
        [s, idx] = sort(s, 'descend'); % Sort eigenvalues in descending order
        U = U(:, idx); % Sort eigenvectors correspondingly
        
        tv = sum(s); % Total variance
        per = s / tv; % Explained variance proportion
        tper = cumsum(per); % Cumulative explained variance
        
        P = U; % Principal components (eigenvectors)
        figure;
        plot(tper, 'o-', 'LineWidth', 1.5);
        title('Cumulative Explained Variance (Eigen Decomposition)');
        xlabel('Number of Components');
        ylabel('Variance Explained');
        grid on;

    else
        error('Invalid type. Use 1 for Eigen decomposition or 2 for SVD.');
    end
    
    % Project data onto principal components
    X_new = P' * data;
end