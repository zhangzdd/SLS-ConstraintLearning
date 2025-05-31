% Define parameters
n_block = 2; % Each block is a 2x1 vector
N = 3;       % We have 3 blocks stacked together

% Create the block downshift operator matrix
D = createBlockDownshiftOperator(n_block, N);

disp('Block Downshift Operator Matrix D:');
disp(full(D)); % Use full() to display the matrix if it's sparse

% Create an example stacked vector X
x1 = [10; 11];
x2 = [20; 21];
x3 = [30; 31];
X = [x1; x2; x3];

disp('Original stacked vector X:');
disp(X);

% Apply the downshift operation
X_shifted = D * X;

disp('Shifted vector X_shifted (D * X):');
disp(X_shifted);

% Expected output for X_shifted:
% [20; 21; 30; 31; 0; 0]