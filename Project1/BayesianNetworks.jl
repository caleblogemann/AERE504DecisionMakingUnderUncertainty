using CSV, DataFrames, Distributions, BayesNets, SpecialFunctions, LightGraphs
using TikzGraphs, TikzPictures, Random

function BayesianScore(graph, df)
    BayesNets.bayesian_score(graph, names(df), df)
end

function local_search(scoring_function, graph_0)
    g = graph_0;
    max_score = scoring_function(g);
    nVariables = nv(g);
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

function k2search(scoring_function, variables, max_parents)
    nVariables = size(variables, 1);
    g = SimpleDiGraph(nVariables);
    max_score = scoring_function(g);
    for i in 1:nVariables
        child = variables[i];
        has_updated = true;
        nParentsAdded = 0;
        while (has_updated && nParentsAdded < max_parents)
            has_updated = false;
            max_new_parent = 0;
            for j in 1:i-1
                parent = variables[j]
                if (!has_edge(g, parent, child))
                    add_edge!(g, parent, child);
                    fg = scoring_function(g);
                    if(fg > max_score)
                        has_updated = true;
                        max_new_parent = parent;
                        max_score = fg;
                    end
                    rem_edge!(g, parent, child);
                end
            end
            if (has_updated)
                # update g
                add_edge!(g, max_new_parent, child)
                nParentsAdded = nParentsAdded+1;
            end
        end
    end
    return (g, max_score);
end

# search function using k2 search and local search with some
# random restarting
function full_search(scoring_function, nRestarts, nVariables)
    g = SimpleDiGraph(nVariables);
    max_score = scoring_function(g);
    n_updates = 0;
    for n = 1:nRestarts
        # randomly shuffle variable order, give different results in k2 search
        variables = shuffle(1:nVariables)
        #println(variables);
        max_parents = convert(Int64, floor(n/(nRestarts/10) + 1))
        #max_parents = min(10, nVariables);
        #println(max_parents)
        #max_parents = 5;
        (temp_g, temp_max_score) = k2search(scoring_function, variables, max_parents);
        (temp2_g, temp2_max_score) = local_search(scoring_function, temp_g);
        if(temp2_max_score > max_score)
            n_updates = n_updates+1;
            println(n)
            max_score = temp2_max_score;
            g = temp2_g;
        end
    end
    #println(n_updates);
    return (g, max_score);
end
