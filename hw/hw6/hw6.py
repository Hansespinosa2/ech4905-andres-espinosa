# Define the data
scenarios = [
    {"probability": 0.025, "gamma_hot": 0.1, "gamma_cold": 3},
    {"probability": 0.05, "gamma_hot": 0.1, "gamma_cold": 10},
    {"probability": 0.1, "gamma_hot": 0.1, "gamma_cold": 34},
    {"probability": 0.15, "gamma_hot": 1.3, "gamma_cold": 3},
    {"probability": 0.35, "gamma_hot": 1.3, "gamma_cold": 10},
    {"probability": 0.15, "gamma_hot": 1.3, "gamma_cold": 34},
    {"probability": 0.1, "gamma_hot": 3, "gamma_cold": 3},
    {"probability": 0.05, "gamma_hot": 3, "gamma_cold": 10},
    {"probability": 0.025, "gamma_hot": 3, "gamma_cold": 34},
]

# Calculate the summation
summation = sum(
    scenario["probability"] * (scenario["gamma_hot"] + scenario["gamma_cold"])
    for scenario in scenarios
)

# Print the result
print(f"The summation of the gammas times the probability is: {summation:.3f}")