parameters1 = {"beta_e": 20,
              "beta_i": 40,
              "theta_e": 0.05,
              "theta_i": 0.05}

parameters2 = {"beta_e": 40,
              "beta_i": 20,
              "theta_e": 0.05,
              "theta_i": 0.05}

set_A_parameters = {
                    "beta_e": 20,
                    "beta_i": 30,
                    "theta_e": 0.1,
                    "theta_i": 0.12,
                    }

set_B_parameters = {
                    "beta_e": 5,
                    "beta_i": 10,
                    "theta_e": 0.05,
                    "theta_i": 0.10,
                    }

sigma = {
        "sigma_ee": 0.35,
        "sigma_ei": 0.48,
        "sigma_ie": 0.60,
        "sigma_ii": 0.69
        }

sigma2 = {
        "sigma_ee": 1,
        "sigma_ei": 1,
        "sigma_ie": 3,
        "sigma_ii": 3
        }

sigma3 = {
        "sigma_ee": 1/3,
        "sigma_ei": 1/3,
        "sigma_ie": 1,
        "sigma_ii": 1
        }
 
normalization = {
                "nu_ee": 1.,
                "nu_ei": 1.,
                "nu_ie": 1.,
                "nu_ii": 1.
                }

weights_1 = {
            "nu_ee": 0.72, # 0.24
            "nu_ei": 0.78, # 0.26
            "nu_ie": 1.,   # 0.33
            "nu_ii": 0.51, # 0.17
            }

weights_2 = {
            "nu_ee": 0.675, # 0.27
            "nu_ei": 0.675, # 0.27
            "nu_ie": 1.,   # 0.4
            "nu_ii": 0.175, # 0.07
            }

set_A_parameters = set_A_parameters | sigma
set_B_parameters = set_B_parameters | sigma
parameters2 = parameters2 | sigma
parameters1 = parameters1 | sigma

set_A_parameters_sigma2 = set_A_parameters | sigma2
set_B_parameters_sigma2 = set_B_parameters | sigma2
parameters2_sigma2 = parameters2 | sigma2
parameters1_sigma2 = parameters1 | sigma2

weights_Ronja = {
            "nu_ee": 2.2, # 0.24
            "nu_ei": 3.3, # 0.26
            "nu_ie": 2.6,   # 0.33
            "nu_ii": 0.9, # 0.17
            }

set_A_Ronja = {
                "beta_e": 5,
                "beta_i": 5,
                "theta_e": 0.,
                "theta_i": 0.,
                }

set_B_Ronja = {
                "beta_e": 5,
                "beta_i": 5,
                "theta_e": 0.05,
                "theta_i": 0.1,
                }

set_C_Ronja = {
                "beta_e": 5,
                "beta_i": 10,
                "theta_e": 0.05,
                "theta_i": 0.1,
                }


param_Ronja = set_A_Ronja | weights_Ronja | sigma2
param_Ronja2 = set_B_Ronja | weights_Ronja | sigma2