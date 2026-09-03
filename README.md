# Continuous Multi-Attribute Recognition for Agent Behaviour Understanding

Source code for the IJCAI-26 paper *Continuous Multi-Attribute Recognition for Agent Behaviour Understanding*,
by Sheryl Mantik, Michael Dann, Huong Ha, Minyi Li, and Julie Porteous.

Code is adapted from *Multi-Agent Intention Recognition and Progression*, by Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, Felipe Meneguzzi and John Thangarajah.

Feel free to email Sheryl at sheryl.mantik@student.rmit.edu.au if you have any issues getting the code to run.

## 1. Installing the Requirements

Code is tested on Python 3.9.4.
Required packages can be found under 'requirements.txt'.

## 2. Model Training

To train a new RL policy, run:

```python python_agent.py train```

The goal items for the RL policy can be configured in scenario.py (line 32).

Agent scores are automatically logged to mod/*flags_during_training*/training_scores.csv.
Flags during training are the ones specified when running the python script.

## 3. Model Evaluation (Based on Experiment \[Section 4\] )

1. For each attribute, we have different setup tied to it. See the 3A for Preferences and 3B for Ability Level.

2. Run this command through terminal:
```python python_agent.py eval multivar uvfa pref ability craft_loc AR modbi```
See Additional Notes below for flag explanation.

### 3A. Preferences
We set which type of agent we want to observe by adjusting scenario.py, and adding the items listed in scenarios[”eval”][”attr_sets”]. We provide the weights we used in our experiments in each attribute. An example how the content of the list should look like:

```python
scenarios["eval"]["attr_sets"] = [
    {"cloth": 0.9, "stick": 0.1}
]
```
Choose 1 weight to be used in scenario.py before running the experiment:
```python
{"cloth": A, "stick": B, "plank": C, "rope": D}
```
All values (A,B,C,D) need to be substituted to value of 0-1 that sums up to one. Values used in the experiments:
- {"cloth": 0.7, "stick": 0.1, "plank": 0.1, "rope": 0.1} -> Strong cloth preference
- {"cloth": 0.1, "stick": 0.1, "plank": 0.1, "rope": 0.7} -> Strong rope preference

### 3B. Ability Level
*TODO: add explanation

### Additional Notes
*TODO: add explanation for flags