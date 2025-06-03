function D = createBlockDownshiftOperator(n_block, N)
% CREATEBLOCKDOWNSHIFTOPERATOR Creates a block downshift operator matrix.
%
%   D = CREATEBLOCKDOWNSHIFTOPERATOR(n_block, N) generates a matrix D that,
%   when multiplied by a stacked vector X of N blocks (each of size n_block),
%   shifts the blocks downwards. The operation is:
%   X_shifted = D * X
%   where X_shifted will have the first block replaced by zeros, and the
%   subsequent blocks shifted up, with a zero block at the end.
%
%   Inputs:
%     n_block - The dimension (size) of each individual block vector.
%     N       - The total number of blocks in the stacked vector.
%
%   Output:
%     D       - The (N*n_block) x (N*n_block) block downshift operator matrix.

    % Total dimension of the stacked vector
    total_dim = N * n_block;

    % Initialize the matrix D with zeros. Using sparse matrix for efficiency
    % if N*n_block is large, as most elements will be zero.
    D = sparse(total_dim, total_dim);

    % Populate the matrix.
    % The i-th block of the output vector (X_shifted) should be the (i+1)-th
    % block of the input vector (X).
    % This means we are mapping rows corresponding to block i in D to columns
    % corresponding to block i+1 in the input.

    for i = 1:(N - 1)
        % Define the row indices for the current output block (i-th block of X_shifted)
        row_start = i * n_block + 1;
        row_end = (i + 1) * n_block;

        % Define the column indices for the corresponding input block ((i+1)-th block of X)
        col_start = (i - 1) * n_block + 1;
        col_end = i * n_block;

        % Place an identity matrix in the appropriate block position
        D(row_start:row_end, col_start:col_end) = eye(n_block);
    end

    % For the last block of the output (N-th block of X_shifted), it will be
    % filled with zeros by default from the sparse initialization.
end