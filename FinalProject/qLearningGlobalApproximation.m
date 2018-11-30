function qLearningGlobalApproximation(dataSet, learningRate, discountFactor, nFareClasses, fareClassPrices)
    n = length(dataSet);
    nFareClasses = 3;

    % stateVector = {time, fareClassToAdd, 
    oldState = struct();
    oldState.time = dataSet(1, 1);
    oldState.nextClass = dataSet(1, 2);
    oldState.currentlyBooked = zeros(nFareClasses,1);
    oldState.bookingTimes = cell(nFareClasses,1);
    % cancellations
    % [timeOfCancellation, class, timeOfBooking]
    cancellations = [];

    for i = 1:n
        % choose action
        % TODO: Add exploration strategy
        accept = 1.0;

        % update State based on action
        newState = oldState;
        reward = 0;
        if(accept)
            newState.currentlyBooked(oldState.nextClass) = newState.currentlyBooked(oldState.nextClass) + 1;
            %newState.bookingTimes
            if(dataSet(i, 3) > 0)
                cancellations = [cancellations; dataSet(i, 3), oldState.nextClass, oldState.time];
                cancellations = sortrows(cancellations);
            end
            reward = 100;
        end
        newState.time = dataSet(i+1, 1);
        newState.nextClass = dataSet(i+1, 2);
        % remove cancellations from booking that occur before next request
        while (cancellations(1, 1) < newState.time)
            classToCancel = cancellations(1,2);
            newState.currentlyBooked(classToCancel) = newState.currentlyBooked(classToCancel) - 1;
            newState.
        end

        % update weights on neural network
        theta = theta + learningRate*(reward + discountFactor*())

        oldState = newState;
    end
end

