using LinearAlgebra

function valueIteration(T, R, states, actions, discountFactor, tol, maxIterations)
    k = 0;
    U = Dict(s => 0.0 for s in states)
    oldnorm = norm(U)
    while (k <= maxIter)
        k = k+1;
        for s in states
            U[s] = maximum([R[s,a] + discountFactor*sum([T[sp,s,a]*U[sp] for sp in states]) for a in actions])
        end
        newnorm = norm(U);
        if (abs(newnorm - oldnorm) < tol)
            break;
        end
        oldnorm = newnorm;
    end
    return U;
end

function policyExtraction(states, actions, U, T, R)
    policy = Dict();
    for s in states
        maxValue = nothing;
        maxAction = 0;
        for a in actions
            value = R[s,a] + sum([T[sp,s,a].*U[sp] for sp in states]);
            if (maxValue == nothing || value >= maxValue)
                maxValue = value;
                maxAction = a;
            end
        end
        policy[s] = maxAction;
    end
    return policy;
end

function policyExtractionQ(states, actions, Q)
    policy = Dict();
    for s in states
        maxValue = nothing;
        maxAction = 0;
        for a in actions
            value = Q[s, a];
            if (maxValue == nothing || value >= maxValue)
                maxValue = value;
                maxAction = a;
            end
        end
        policy[s] = maxAction;
    end
    return policy;
end

function qLearning(states, actions, data, learningRate, discountFactor)
    t = 0;
    numRows = size(data,1);
    Q = Dict((s, a) => 0.0 for s in states, a in actions);
    for i = 1:numRows
        s = df[i,:s];
        a = df[i,:a];
        r = df[i,:r];
        sp = df[i,:sp];
        Q[s, a] = Q[s, a] + learningRate*(r + discountFactor*maximum([Q[sp, ap] for ap in actions]) - Q[s, a]);
        t = t+1;
    end
    return Q;
end

function sarsa(states, actions, data, learningRate, discountFactor)
    t = 0;
    numRows = size(data,1);
    Q = Dict((s, a) => 0.0 for s in states, a in actions);
    for i = 1:numRows-1
        s = df[i,:s];
        at = df[i,:a];
        atp1 = df[i+1,:a];
        r = df[i,:r];
        sp = df[i,:sp];
        Q[s, at] = Q[s, at] + learningRate*(r + discountFactor*Q[sp, atp1] - Q[s, at]);
        t = t+1;
    end
    return Q;
end
