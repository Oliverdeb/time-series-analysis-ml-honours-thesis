"""Page-Hinkley"""


class PageHinkley:
    def __init__(self, delta_=0.005, lambda_=50, alpha_=1 - 0.0001):
        self.delta_ = delta_
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.m_n = 1
        self.sum = 0.0
        # incrementally calculated mean of input data
        self.x_mean = 0.0
        # number of considered values
        self.num = 0.0
        self.change_detected = False

    def __reset_params(self,delta_=0.005, lambda_=50, alpha_=1 - 0.0001):
        """
        Every time a change has been detected, all the collected statistics are reset.
        :return:
        """
        self.delta_ = delta_
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.m_n = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def set_input(self, x):
        """
        Main method for adding a new data value and automatically detect a possible concept drift.
        :param x: input data
        :return: boolean
        """
        self.__detect_drift(x)

        return self.change_detected

    def __detect_drift(self, x):
        """
        Concept drift detection following the formula from 'Knowledge Discovery from Data Streams' by João Gamma (p. 76)
        :param x: input data
        Follows PageHinkley algorithm for drift detection from the MOA git
        """
        # calculate the average and sum
        self.x_mean = self.x_mean + (x - self.x_mean) / self.m_n;
        self.sum = self.sum * self.alpha_ + (x - self.x_mean - self.delta_)
        self.m_n += 1
        self.change_detected = True if self.sum > self.lambda_ else False
        if self.change_detected:
            print("The sum was: ",self.sum)
            self.__reset_params()
