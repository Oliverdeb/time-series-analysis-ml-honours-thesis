"""EDDM"""
import math

class EDDM:
    def __init__(self):
        self.FDDM_OUTCONTROL = 0.9
        self.FDDM_WARNING = 0.95
        self.FDDM_MINNUMINSTANCES = 30
        self.m_numErrors = 0
        self.m_minNumErrors = 30
        self.m_n = 1
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_stdTemp = 0.0
        self.m_m2smax = 0.0
        self.change_detected = False
        self.isWarningZone = False
        self.estimation = 0.0
        self.delay = 0



    def __reset_params(self):
        """
        Every time a change has been detected, all the collected statistics are reset.
        :return:
        """
        self.m_numErrors = 0
        self.m_minNumErrors = 30
        self.m_n = 1
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_stdTemp = 0.0
        self.m_m2smax = 0.0
        self.change_detected = False
        self.isWarningZone = False
        self.estimation = 0.0

    def set_input(self, x):
        """
        Main method for adding a new data value and automatically detect a possible concept drift.
        :param x: input data
        :return: boolean
        """
        self.__detect_drift(x)
        #raw_input()
        return self.change_detected

    def __detect_drift(self, x):
        """
        :param x: input data
        """
        self.m_n +=1
        if x == 1.0:
            self.isWarningZone = False
            self.delay = 0
            self.m_numErrors +=1
            self.m_lastd = self.m_d
            self.m_d = self.m_n -1
            self.distance = self.m_d - self.m_lastd
            self.oldmean = self.m_mean
            self.m_mean = self.m_mean + (self.distance - self.m_mean)/self.m_numErrors
            self.estimation = self.m_mean
            self.m_stdTemp = self.m_stdTemp + (self.distance - self.m_mean) * (self.distance - self.oldmean)
            self.std = math.sqrt(self.m_stdTemp / self.m_numErrors);
            self.m2s = self.m_mean + 2 * self.std
            if self.m2s > self.m_m2smax:
                if self.m_n > self.FDDM_MINNUMINSTANCES:
                    self.m_m2smax = self.m2s
            else:
                self.p = self.m2s/self.m_m2smax
                if (self.m_n > self.FDDM_MINNUMINSTANCES and self.m_numErrors > self.m_minNumErrors and self.p < self.FDDM_OUTCONTROL):
                    self.change_detected = True;
                elif (self.m_n > self.FDDM_MINNUMINSTANCES and self.m_numErrors > self.m_minNumErrors and self.p < self.FDDM_WARNING):
                    self.isWarningZone = True
                else:
                    self.isWarningZone = False

        if self.change_detected:
            self.__reset_params()
