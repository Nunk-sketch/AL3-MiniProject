# You can assume that there is no direct link between the variables confounded by _H
# Interventions will be clamped between -2 and 2
# _H is a confounder (not a mediator or collider) for exactly two variables
# In total 7 nodes where _H is not observed
TOTAL_NODES = 7
COST_PER_SAMPLE = 1
COST_PER_INCORRECT_GUESS = 70
COST_PER_EXPERIMENT = 20 # query


def main():
    print("Hello from al3-miniproject!")


if __name__ == "__main__":
    main()
