# UAMPref
Companion code for S. M. Katz, A. LeBihan, and M. J. Kochenderfer, “Learning an urban air mobility encounter model from expert preferences,” in Digital Avionics Systems Conference (DASC), 2019

## File Descriptions
`reward_iteration.jl` - Main file for algorithm implementation. Simply include this file and then run `reward_iteration(num_iter)`.  
`ri_functions.jl` - Contains all functions called by the reward iteration algorithm.  
`ri_const.jl` - Defines constants used in reward iteration algorithm.  
`ri_types.jl` - Defines types used in reward iteration algorithm.  
`landing_mdp.jl` - Defines landing Markov Decision Process (MDP) using POMDPs.jl.  
`ri_test.jl` - Contains functions and scripts to test algorithm performance automatically.  

### Interactive Version for Jupyter Notebooks (IJulia)
`interactive_reward_iteration.jl` - Main file for algorithm implementation. Include this file in Jupyter notebook rather than `reward_iteration.jl`.  
`InteractivePreferences.ipynb` - Example Jupyter notebook for running the interactive version.  
`support_code.jl` - Code used for plotting. From https://github.com/sisl/algforopt-notebooks.  
