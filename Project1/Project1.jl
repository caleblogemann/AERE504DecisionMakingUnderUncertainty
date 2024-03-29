using CSV, DataFrames, Distributions, BayesNets, SpecialFunctions, LightGraphs
using TikzGraphs, TikzPictures, Printf

include("BayesianNetworks.jl")

function compute_r(dt, nVariables)
    r = zeros(Int64, nVariables);
    for i in 1:nVariables
        r[i] = maximum(dt[:,i]) - minimum(dt[:,i]) + 1;
    end
    return r;
end

function save_graph(g, filename, variables)
    filename = filename*".pdf"
    t = plot(g, map(string, variables));
    save(PDF(filename), t);
end

function save_graph_file(g, filename, variables)
    filename = filename*".gph"
    file = open(filename, "w")

    for edge in LightGraphs.edges(g)
        @printf(file,"%s,%s\n",variables[LightGraphs.src(edge)], variables[LightGraphs.dst(edge)])
    end

    close(file)
end

titanicData = CSV.read("titanic.csv");
whitewineData = CSV.read("whitewine.csv");
schoolgradesData = CSV.read("schoolgrades.csv");
structuredlearningData = CSV.read("structurelearning_test.csv");

dfArray = [titanicData, whitewineData, schoolgradesData, structuredlearningData];
nRestartsArray = [1000, 1000, 500, 1000];
filenames = ["titanic", "whitewine", "schoolgrades", "structurelearning_test"]

for i = 1:4
    df = dfArray[i];
    nRestarts = nRestartsArray[i];
    dt = convert(Array, df);
    nVariables = size(dt, 2);
    r = compute_r(dt, nVariables);
    @time (g, max_score) = full_search(g->BayesianScore(g, df), nRestarts, nVariables);
    println(filenames[i]);
    println(max_score);
    save_graph(g, filenames[i], names(df));
    save_graph_file(g, filenames[i], names(df));
end


