from utils.model_estimation.copula.student_estimation import StudentCopulaVaR
from utils.model_estimation.copula.gaussian_estimation import GaussianCopulaVaR
from utils.model_estimation.copula.plackett_estimation import PlackettCopulaVaR
from utils.model_estimation.model.msm_estimation import MSMEstimation
from utils.model_estimation.model.garch_estimation import GarchEstimation
from utils.model_estimation.model.mean_reverting_estimation import MeanRevertingEstimation


class ValueAtRiskCalculationFactory:
    @staticmethod
    def create_var_calculator(copula_type, estimation_type):
        if estimation_type == 'msm' and copula_type == 'student':
            return StudentCopulaVaR(MSMEstimation())
        elif estimation_type == 'garch' and copula_type == 'student':
            return StudentCopulaVaR(GarchEstimation())
        elif estimation_type == 'mean_reverting' and copula_type == 'student':
            return StudentCopulaVaR(MeanRevertingEstimation())
        elif estimation_type == 'msm' and copula_type == 'gaussian':
            return GaussianCopulaVaR(MSMEstimation())
        elif estimation_type == 'garch' and copula_type == 'gaussian':
            return GaussianCopulaVaR(GarchEstimation())
        elif estimation_type == 'mean_reverting' and copula_type == 'gaussian':
            return PlackettCopulaVaR(MeanRevertingEstimation())
        elif estimation_type == 'msm' and copula_type == 'plackett':
            return PlackettCopulaVaR(MSMEstimation())
        elif estimation_type == 'garch' and copula_type == 'plackett':
            return PlackettCopulaVaR(GarchEstimation())
        elif estimation_type == 'mean_reverting' and copula_type == 'plackett':
            return PlackettCopulaVaR(MeanRevertingEstimation())
        else:
            raise ValueError("Unsupported estimation type.")
