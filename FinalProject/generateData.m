% clear so only variables created now are saved
clear;
close all;
% Parameters
nFareClasses = 3;
capacity = 100;
classSizeMean = [10, 30, 60];
classCancelRate = [0.05, 0.02, 0.03];
totalTime = 1000;
fareClassPrices = [300, 200, 100];

nDataSets = 100000;
fileName = 'test';


% sample sizes
%classSizeSD = [5, 5, 10];
% classSizes(Data Set Index, Fare Class Index)
% Assume poisson
%classSizes = cell2mat(arrayfun(@(j) normrnd(classSizeMean(j), classSizeSD(j), 1, nDataSets)', 1:nFareClasses, 'UniformOutput', false));
% if negative generated force to 0
%classSizes(classSizes<0) = 0;

% Assume Poisson process with mean number of arrivals is classSizeMean
% TODO: could add step such that classSizeMean varies according to normal distribution for each data set
averageInterArrivalTime = totalTime./classSizeMean;
% TODO: This could get large as nDataSets increases
% TODO: May need to increase max number of times to insure that totalTime is reached
maxNumberOfArrivals = poissinv(1-1/(10*nDataSets), classSizeMean);

% get cell array of arrival times for a single fare class
% cell array necessary allow for different number of arrivals with each flight
listArrivalTimes = @(i) num2cell(cumsum(exprnd(averageInterArrivalTime(i), nDataSets, maxNumberOfArrivals(i)), 2), 2);

% arrivalTimes{Data Set Index, Fare Class Index}(arrival Index), arrival Index < nArrivals
arrivalTimes = arrayfun(@(i) listArrivalTimes(i) , 1:nFareClasses, 'UniformOutput', false);
arrivalTimes = horzcat(arrivalTimes{:});
% cut extra arrival times after totalTime
arrivalTimes = cellfun(@(c) c(c < totalTime), arrivalTimes, 'UniformOutput', false);

% nArrivals(Data Set Index, Fare Class Index)
nArrivals = cellfun(@length, arrivalTimes);

% cancellations{Data Set Index, Fare Class Index}(arrival Index)
% zero for not cancelled otherwise uniform between zero and one for percentage
% of time left before cancellation
classCancellations = @(j) arrayfun(@(i) binornd(1, classCancelRate(j), 1, nArrivals(i, j)).*rand(1, nArrivals(i, j)), 1:nDataSets, 'UniformOutput', false)';
cancellations = arrayfun(@(i) classCancellations(i), 1:nFareClasses, 'UniformOutput', false);
cancellations = horzcat(cancellations{:});

% nCancellations(Data Set Index, Fare Class Index)
nCancellations = cellfun(@(c) sum(ceil(c)), cancellations)
% maxReward(Data Set Index)
optimalNumberBookings = max(nArrivals - nCancellations, capacity*ones(size(nArrivals)))
maxReward = nArrivals(:,1) - nCancellations(:,1)

% dataSets{i} = [t, fareClassIndex, timeOfCancellation]
% [1, 1, 0] - adding passenger of class 1 at t = 1
% [2, 3, 1] - adding passenger of class 3 at t = 2, will cancel at some later time
% ...
computeCancellationTime = @(j, i) ((totalTime-arrivalTimes{j,i}).*cancellations{j,i} + arrivalTimes{j,i}.*ceil(cancellations{j,i}))';
computeDataSet = @(j) cell2mat(arrayfun(@(i) [arrivalTimes{j, i}', i*ones(nArrivals(j, i), 1), computeCancellationTime(j, i)], 1:nFareClasses, 'UniformOutput', false)');
dataSets = arrayfun(@(j) computeDataSet(j), 1:nDataSets, 'UniformOutput', false);
% sort by timestamp
dataSets = cellfun(@sortrows, dataSets, 'UniformOutput', false);



% saveall
save(strcat(fileName, '.mat'))
