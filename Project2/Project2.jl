using CSV, DataFrames, Printf

df = CSV.read("transitions.csv");
states = unique(df[:s]);
actions = unique(df[:a]);
numRows = size(df,1);
counts = Dict((s, a, sp) => 0 for s in states, a in actions, sp in states);
rewardsum = Dict((s, a) => 0.0 for s in states, a in actions);

for i = 1:numRows
    s = df[i,1];
    a = df[i,2];
    r = df[i,3];
    sp = df[i,4];

    counts[s, a, sp] = counts[s, a, sp] + 1;
    rewardsum[s, a] = rewardsum[s, a] + r;
end

T = Dict((sp, s, a) => 0.0 for s in states, a in actions, sp in states);
R = Dict((s, a) => 0.0 for s in states, a in actions);

for s in states
    for a in actions
        countSum = sum([counts[s, a, sp] for sp in states]);
        if(countSum != 0)
            R[s, a] = rewardsum[s, a]/countSum;
            for sp in states
                T[sp, s, a] = counts[s, a, sp]/countSum;
            end
        end
    end
end

include("ReinforcementLearning.jl")
# Value Iteration
discountFactor = 0.95;
tol = 1e-8;
maxIter = 100;
U = valueIteration(T, R, states, actions, discountFactor, tol, maxIter);
policy = policyExtraction(states, actions, U, T, R);

# write policy to file
filename = "transitions.policy";
file = open(filename, "w");
for i = 1:size(states,1)
    @printf(file, "%i\n", policy[i]);
end
close(file);

# Qlearning
#learningRate = 0.5;
#Q = qLearning(states, actions, df, learningRate, discountFactor);
#policy2 = policyExtractionQ(states, actions, Q);
# SARSA
#Q = sarsa(states, actions, df, learningRate, discountFactor);
#policy3 = policyExtractionQ(states, actions, Q);
