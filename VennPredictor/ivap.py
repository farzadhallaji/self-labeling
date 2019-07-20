from VennPredictor.util import *


class ivap:

    def __init__(self, base_classifier, train_x, train_y, calibration_x, calibration_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.calibration_x = calibration_x
        self.calibration_y = calibration_y
        self.test_x = test_x
        self.kPrime = None
        self.base_classifier = base_classifier

    def algorithm1(self, P):

        S = []
        P[-1] = np.array((-1, -1))
        push(P[-1], S)
        push(P[0], S)
        for i in range(1, self.kPrime + 1):
            while len(S) > 1 and nonleftTurn(nextToTop(S), top(S), P[i]):
                pop(S)
            push(P[i], S)
        return S

    def algorithm2(self, P, S):

        Sprime = S[::-1]  # reverse the stack

        F1 = np.zeros((self.kPrime + 1,))
        for i in range(1, self.kPrime + 1):
            F1[i] = slope(top(Sprime), nextToTop(Sprime))
            P[i - 1] = P[i - 2] + P[i] - P[i - 1]
            if notBelow(P[i - 1], top(Sprime), nextToTop(Sprime)):
                continue
            pop(Sprime)
            while len(Sprime) > 1 and nonleftTurn(P[i - 1], top(Sprime), nextToTop(Sprime)):
                pop(Sprime)
            push(P[i - 1], Sprime)
        return F1

    def algorithm3(self, P):

        P[self.kPrime + 1] = P[self.kPrime] + np.array((1.0, 0.0))

        S = []
        push(P[self.kPrime + 1], S)
        push(P[self.kPrime], S)
        for i in range(self.kPrime - 1, 0 - 1, -1):  # k'-1,k'-2,...,0
            while len(S) > 1 and nonrightTurn(nextToTop(S), top(S), P[i]):
                pop(S)
            push(P[i], S)
        return S

    def algorithm4(self, P, S):

        Sprime = S[::-1]  # reverse the stack

        F0 = np.zeros((self.kPrime + 1,))
        for i in range(self.kPrime, 1 - 1, -1):  # k',k'-1,...,1
            F0[i] = slope(top(Sprime), nextToTop(Sprime))
            P[i] = P[i - 1] + P[i + 1] - P[i]
            if notBelow(P[i], top(Sprime), nextToTop(Sprime)):
                continue
            pop(Sprime)
            while len(Sprime) > 1 and nonrightTurn(P[i], top(Sprime), nextToTop(Sprime)):
                pop(Sprime)
            push(P[i], Sprime)
        return F0[1:]

    def prepareData(self, calibrPoints):

        ptsSorted = sorted(calibrPoints)

        xs = np.fromiter((p[0] for p in ptsSorted), float)
        ys = np.fromiter((p[1] for p in ptsSorted), float)
        ptsUnique, ptsIndex, ptsInverse, ptsCounts = np.unique(xs,
                                                               return_index=True,
                                                               return_counts=True,
                                                               return_inverse=True)
        a = np.zeros(ptsUnique.shape)
        np.add.at(a, ptsInverse, ys)
        # now a contains the sums of ys for each unique value of the objects

        w = ptsCounts
        yPrime = a / w
        yCsd = np.cumsum(w * yPrime)  # Might as well do just np.cumsum(a)
        xPrime = np.cumsum(w)
        self.kPrime = len(xPrime)

        return yPrime, yCsd, xPrime, ptsUnique

    def computeF(self, xPrime, yCsd):
        P = {0: np.array((0, 0))}
        P.update({i + 1: np.array((k, v)) for i, (k, v) in enumerate(zip(xPrime, yCsd))})

        S = self.algorithm1(P)
        F1 = self.algorithm2(P, S)

        # P = {}
        # P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})

        S = self.algorithm3(P)
        F0 = self.algorithm4(P, S)

        return F0, F1

    def getFVal(self, F0, F1, ptsUnique, testObjects):
        pos0 = np.searchsorted(ptsUnique[1:], testObjects, side='right')
        pos1 = np.searchsorted(ptsUnique[:-1], testObjects, side='left') + 1
        return F0[pos0], F1[pos1]

    def ScoresToMultiProbs(self, calibrPoints, testObjects):
        # sort the points, transform into unique objects, with weights and updated values
        yPrime, yCsd, xPrime, ptsUnique = self.prepareData(calibrPoints)

        # compute the F0 and F1 functions from the CSD
        F0, F1 = self.computeF(xPrime, yCsd)

        # compute the values for the given test objects
        p0, p1 = self.getFVal(F0, F1, ptsUnique, testObjects)

        return p0, p1

    def computeF1(self, yCsd, xPrime):

        P = {0: np.array((0, 0))}
        P.update({i + 1: np.array((k, v)) for i, (k, v) in enumerate(zip(xPrime, yCsd))})

        S = self.algorithm1(P)
        F1 = self.algorithm2(P, S)

        return F1

    def ScoresToMultiProbsV2(self, calibrPoints, testObjects):
        # sort the points, transform into unique objects, with weights and updated values
        yPrime, yCsd, xPrime, ptsUnique = self.prepareData(calibrPoints)

        # compute the F0 and F1 functions from the CSD
        F1 = self.computeF1(yCsd, xPrime)
        pos1 = np.searchsorted(ptsUnique[:-1], testObjects, side='left') + 1
        p1 = F1[pos1]

        yPrime, yCsd, xPrime, ptsUnique = self.prepareData((-x, 1 - y) for x, y in calibrPoints)
        F0 = 1 - self.computeF1(yCsd, xPrime)
        pos0 = np.searchsorted(ptsUnique[:-1], testObjects, side='left') + 1
        p0 = F0[pos0]

        return p0, p1

    def confidenceV1(self):
        model = self.base_classifier
        model.fit(self.train_x, self.train_y)
        calibrScores = np.max(model.predict_proba(self.calibration_x), axis=1)
        calibrPoints = [(score, label) for score, label in zip(calibrScores, self.calibration_y)]
        testObjects = np.max(model.predict_proba(self.test_x), axis=1)

        return self.ScoresToMultiProbs(calibrPoints, testObjects)

    def confidenceV2(self):
        model = self.base_classifier
        model.fit(self.train_x, self.train_y)
        calibrScores = np.max(model.predict_proba(self.calibration_x), axis=1)
        calibrPoints = [(score, label) for score, label in zip(calibrScores, self.calibration_y)]
        testObjects = np.max(model.predict_proba(self.test_x), axis=1)

        return self.ScoresToMultiProbsV2(calibrPoints, testObjects)
