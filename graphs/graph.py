import abc


class GraphStructure:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def define_SEM():
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def fit_all_models(self):

        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def refit_models(self, observational_samples):
        raise NotImplementedError("Subclass should implement this.")


    @abc.abstractmethod
    def get_all_do(self):
        raise NotImplementedError("Subclass should implement this.")

