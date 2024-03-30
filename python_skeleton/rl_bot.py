
class ApproximateQAgent():
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self):
        self.weights = np.zeros()
        pass

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator

          state: represents the current state
          action: represents action taken
          return: float, representing Q_w(state,action)
        """
        features = self.featExtractor.getFeatures(state, action)
        if len(self.getWeights()) == 0:
            for feature in features:
                self.weights[feature] = 0

        "*** YOUR CODE HERE ***"
        total = 0
        for feature in features:
            total += features[feature] * self.weights[feature]
        return total

    def update(self, state, action, nextState, reward):
        """
          Should update your weights based on transition

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)

        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass