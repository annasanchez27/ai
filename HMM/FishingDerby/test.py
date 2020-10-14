def guess(self, step, observations):
    """
    This method gets called on every iteration, providing observations.
    Here the player should process and store this information,
    and optionally make a guess by returning a tuple containing the fish index and the guess.
    :param step: iteration number
    :param observations: a list of N_FISH observations, encoded as integers
    :return: None or a tuple (fish_id, fish_type)
    """
    # This code would make a random guess on each step:
    # return (step % N_FISH, random.randint(0, N_SPECIES - 1))
    print("STEP", step)
    T = 10
    if (step <= T):
        self.Obs.append(observations)
    if (step == T):
        self.Obs = self.transpose(self.Obs)

        model1 = HMM3()
        newA, newB, newpi = model1.run_learningalgorithm(self.A, self.B, self.pi, self.Obs[0])
        self.modelfish[self.currentfish] = [newA, newB, newpi]
        return (self.currentfish, 0)

    if (step > T):
        probofobs = [None for i in range(N_SPECIES)]
        for model, i in zip(self.modelspecies, range(N_SPECIES)):
            if model != None:
                f = HMM1()
                prob = f.forward_algorithm(model[0], model[1], model[2], self.Obs[1])
                probofobs[i] = prob

        indexspecies = probofobs.index(max(probofobs))

        return (1, indexspecies)

    return None
