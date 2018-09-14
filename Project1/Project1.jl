using CSV, DataFrame, Distributions, BayesNets

titanicData = CSV.read("titanic.csv");
df = titanicData;
nDataPoints = size(titanicData, 1);
nVariables = size(titanicData, 2);

r = zeros(Int64, nVariables);

for i in 1:nVariables
    r[i] = maximum(df[:,i]) - minimum(df[:,i]) + 1;
end

function get_idx(multi_idx, index_lengths)
    n_indices = size(multi_idx,1);
    idx = (multi_idx[1] - 1);
    for i = 2:n_indices
        idx = idx*index_lengths[i] + multi_idx[i] - 1
    end
    idx = idx + 1;
    return idx
end

function get_multi_idx(idx, index_lengths)
    n_indices = size(index_lengths, 1);
    idx = idx - 1;
    multi_idx = zeros(Int64, n_indices);
    for i = n_indices:-1:1
        n = mod(idx, index_lengths[i]) + 1;
        multi_idx(i) = n;
        idx = (idx - (n - 1))/index_length[i];
    end
end

# Use uniform graph priors
# Two possible parental distribution priors
# uniform alpha_{ijk} = 1
# BDeu alpha_{ijk} = 1/(q_i r_i)
function BayesianScore(graph, df, r)
    nVariables = size(r,1);
    nRows = size(df,1);
    # compute q
    q = ones(Int64, nVariables);
    # compute pi
    nParents = zeros(Int64, nVariables);
    parents = Dict();
    # compute pi
    pi_array = Dict();
    j_array = Dict();
    for i in 1:nVariables
        parents[i] = inneighbors(graph, i);
        nParents[i] = size(parents[i],1);
        for k = 1:nParents[i]
            q[i] = q[i]*r[parents[i][k]];
        end

        for j = 1:q[i]
            pi_array[i,j] = get_multi_idx(j, r[parents[i]]);
            j_array[i, pi_array[i,j]] = j;
        end
    end

    m = zeros(Int64, nVariables, maximum(q), maximum(r))
    # compute m
    for i = 1:nVariables
        for row = 1:nRows
            multi_idx = df[row, parents[i]];
            j = j_array[i, multi_idx];
            k = df[row, i];
            m[i, j, k] = m[i, j, k] + 1;
        end
    end

    # compute m0
    m0 = zeros(Int64, nVariables, maximum(q));
    for i = 1:nVariables
        for j = 1:q[i]
            m0[i, j] = sum(m[i, j, :]);
        end
    end

    # compute alpha
    alpha = zeros(Float64, nVariables, maximum(q), maximum(r));
    chi = 1.0;
    # compute alpha0
    alpha0 = zeros(Float64, nVariables, maximum(q));
    for i = 1:nVariables
        for j = 1:q[i]
            for k = 1:r[i]
                alpha[i, j, k] = chi/(q[i]*r[i]);
            end
            alpha0[i,j] = sum(alpha[i, j, :])
        end
    end

    # compute Bayesian Score
    bs = 0.0;
    for i = 1:nVariables
        for j = 1:q[i]
            bs = bs + lgamma(alpha0[i,j]) - lgamma(alpha0[i,j] + m0[i,j]);
            for k = 1:r[i]
                bs = bs + lgamma(alpha[i, j, k] + m[i, j, k]) - lgamma(alpha[i, j, k]);
            end
        end
    end
    return bs;
end




