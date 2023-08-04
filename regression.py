from sklearn.linear_model import LinearRegression
import numpy as np
from eelbrain_utilities import load_all_trials_features
from f_eelbrain import remove_indeces

# Lets take a model with features F1, F2 .. Fn, for the subject S with electrodes E1, E2, ... Em
# This function should take F1, F2, ... Fn-1 at electrode E1 and predict Fn at electrode E1 for subject s

# Takes in string electrode, string subject, dictionary parameters, dictionary feature (i.e.{POS: True})
def predict_feature(subject, parameters, feature):
    params_variables, parameters_list, feature = prepare_data(subject, parameters, feature)
    print(f"Calculating R-squared for feature {feature} using parameters {parameters_list}")
    print(f"{params_variables[0].shape} ||| {params_variables[1].shape} ||| {feature}")
    # Parameters: each row is a case, each column is a feature
    x = np.hstack(params_variables)

    model = LinearRegression().fit(x, feature)
    print(f"R-squared for feature {feature} is {model.score(params_variables, feature)}")



def prepare_data(subject, parameters, feature):
    pos, prob, phoneme, envelope, envelope_der, pos_onset = load_all_trials_features(subject)
    params = {"POS": pos, "Phoneme": phoneme, "Probability": prob, "Envelope": envelope,
              "Envelope_Derivative": envelope_der, "POS_Onset": pos_onset}
    parameters = [p for p in parameters if parameters[p]]
    feature = [f for f in feature if feature[f]]
    params_list = []
    for key in params:
        if key in parameters:
            params_list.append(params[key])
    return params_list, parameters, feature

# feature must only be one feature long
def ensure_feature_parameter_dont_overlap(parameters, feature):
    for p in parameters:
        if parameters[p] and feature[p]: # if both are true set parameters to false
            parameters[p] = False
    return parameters

if __name__ == "__main__":
    el = "Cz"
    subject = "S18"
    f = ["POS", "Phoneme", "Probability", "Envelope", "Envelope_Derivative", "POS_Onset"]

    parameters = {"POS": True,
                  "Phoneme": False,
                  "Probability": False,
                  "Envelope": True,
                  "Envelope_Derivative": True,
                  "POS_Onset": False}


    feature = {"POS": True}

    for fs in f:
        if fs not in feature:
            feature[fs] = False

    parameters = ensure_feature_parameter_dont_overlap(parameters, feature)

    predict_feature(subject, parameters, feature)