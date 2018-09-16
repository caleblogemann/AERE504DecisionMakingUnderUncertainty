using CSV, DataFrames, Distributions, BayesNets, SpecialFunctions, LightGraphs
using TikzGraphs, TikzPictures

function get_idx(multi_idx, index_lengths)
    n_indices = size(multi_idx,1);
    idx = (multi_idx[1] - 1);
    for i = 2:n_indices
        idx = idx*index_lengths[i] + multi_idx[i] - 1
    end
    idx = idx + 1;
    return idx;
end

function get_multi_idx(idx, index_lengths)
    n_indices = size(index_lengths, 1);
    idx = idx - 1;
    multi_idx = zeros(Int64, n_indices);
    for i = n_indices:-1:1
        n = mod(idx, index_lengths[i]) + 1;
        multi_idx[i] = n;
        idx = (idx - (n - 1))/index_lengths[i];
    end
    return multi_idx;
end

# Use uniform graph priors
# Two possible parental distribution priors
# uniform alpha_{ijk} = 1
# BDeu alpha_{ijk} = 1/(q_i r_i)
function BayesianScore(graph, dt, r)
    nVariables = size(r,1);
    nRows = size(dt,1);
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
            j = 1;
            if (nParents[i] > 0)
                multi_idx = dt[row, parents[i]];
                j = j_array[i, multi_idx];
            end
            k = dt[row, i];
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

function BayesianScore2(graph, df)
    BayesNets.bayesian_score(graph, names(df), df)
end

function local_search(scoring_function, graph_0)
    g = graph_0;
    max_score = scoring_function(g);
    g_max_score = g;
    has_updated = true;
    while (has_updated)
        g = g_max_score;
        has_updated = false;
        for i = 1:nVariables
            for j = i:nVariables
                if (has_edge(g, i, j))
                    # remove edge
                    rem_edge!(g, Edge(i, j));
                    fg = scoring_function(g);
                    if(fg > max_score)
                        has_updated = true;
                        max_score = fg;
                        g_max_score = g;
                    end

                    # try switching edge direction
                    add_edge!(g, Edge(j, i))
                    if(!is_cyclic(g))
                        fg = scoring_function(g);
                        if(fg > max_score)
                            has_updated = true;
                            max_score = fg;
                            g_max_score = g;
                        end
                    end
                    rem_edge!(g, Edge(j, i));
                    add_edge!(g, Edge(i, j));
                elseif (has_edge(g, j, i))
                    # remove edge
                    rem_edge!(g, Edge(j, i))
                    fg = scoring_function(g);
                    if(fg > max_score)
                        has_updated = true;
                        max_score = fg;
                        g_max_score = g;
                    end

                    # try add edge other direction
                    add_edge!(g, Edge(i, j))
                    if(!is_cyclic(g))
                        fg = scoring_function(g);
                        if(fg > max_score)
                            has_updated = true;
                            max_score = fg;
                            g_max_score = g;
                        end
                    end
                    rem_edge!(g, Edge(i, j));
                    add_edge!(g, Edge(j, i));
                else
                    # try adding edge i \to j
                    add_edge!(g, Edge(i, j))
                    if(!is_cyclic(g))
                        fg = scoring_function(g);
                        if(fg > max_score)
                            has_updated = true;
                            max_score = fg;
                            g_max_score = g;
                        end
                    end
                    rem_edge!(g, Edge(i, j));

                    # try adding edge j \to i
                    add_edge!(g, Edge(j, i))
                    if(!is_cyclic(g))
                        fg = scoring_function(g);
                        if(fg > max_score)
                            has_updated = true;
                            max_score = fg;
                            g_max_score = g;
                        end
                    end
                    rem_edge!(g, Edge(j, i));
                end
            end
        end
    end
    return (g_max_score, max_score);
end

function save_graph(g, filename, variables)
    t = plot(g, map(string, variables));
    save(PDF(filename), t);
end

function k2search(scoring_function, variables, max_parents)
    nVariables = size(variables, 1);
    g = SimpleDiGraph(nVariables);
    max_score = scoring_function(g);
    for i = 1:nVariables
        has_updated = true;
        nParentsAdded = 0;
        while (has_updated && nParentsAdded < max_parents)
            has_updated = false;
            max_parent = 0;
            for j = 1:nVariables
                if (j != i && !has_edge(g, j, i))
                    add_edge!(g, j, i);
                    if(!is_cyclic(g))
                        fg = scoring_function(g);
                        if(fg > max_score)
                            has_updated = true;
                            max_parent = j;
                            max_score = fg;
                        end
                    end
                    rem_edge!(g, j, i);
                end
            end
            if (has_updated)
                # update g
                add_edge!(g, max_parent, i)
                nParentsAdded = nParentsAdded+1;
            end
        end
    end
    return (g, max_score);
end

# search function using k2 search and local search with some
# random restarting
function full_search(scoring_function, nVariables)
    g = SimpleDiGraph(nVariables);
    max_score = scoring_function(g);
    n_updates = 0;
    for n = 1:min(nVariables, 10)
        # randomly shuffle variable order, give different results in k2 search
        variables = shuffle(1:nVariables)
        #println(variables);
        #max_parents = convert(Int64, floor(n/(nRestarts/4) + 2))
        max_parents = min(10, nVariables);
        #println(max_parents)
        (temp_g, temp_max_score) = k2search(scoring_function, variables, n);
        (temp2_g, temp2_max_score) = local_search(scoring_function, temp_g);
        if(temp2_max_score > max_score)
            n_updates = n_updates+1;
            println(n)
            max_score = temp2_max_score;
            g = temp2_g;
        end
    end
    println(n_updates);
    return (g, max_score);
end
