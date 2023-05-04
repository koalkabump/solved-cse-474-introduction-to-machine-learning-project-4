Download Link: https://assignmentchef.com/product/solved-cse-474-introduction-to-machine-learning-project-4
<br>



<h1>1        Overview</h1>

Your task is to build a reinforcement learning agent to navigate the classic 4×4 grid-world environment. The agent will learn an optimal policy through Q-Learning which will allow it to take actions to reach a goal while avoiding obstacles. To simplify the task, we will provide the working environment as well as a framework for a learning agent. The environment and agent will be built to be compatible with OpenAI Gym environments and will run effectively on computationally-limited machines. Furthermore, we have provided two agents (random and heuristic agent) as examples.

<h1>2        Background</h1>

<h2>2.1        Reinforcement Learning</h2>

Reinforcement learning is a machine learning paradigm which focuses on how automated agents can learn to take actions in response to the current state of an environment so as to maximize some reward. This is typically modeled as a Markov decision process (MDP), as illustrated in Figure 1.

Figure 1: The canonical MDP diagram<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>

An MDP is a 4-tuple (<em>S,A,P,R</em>), where

<ul>

 <li><em>S </em>is the set of all possible states for the environment</li>

 <li><em>A </em>is the set of all possible actions the agent can take</li>

 <li><em>P </em>= <em>Pr</em>(<em>s<sub>t</sub></em><sub>+1 </sub>= <em>s</em><sup>0</sup>|<em>s<sub>t </sub></em>= <em>s,a<sub>t </sub></em>= <em>a</em>) is the state transition probability function</li>

 <li><em>R </em>: <em>S </em>× <em>A </em>× <em>S </em>→ R is the reward function</li>

</ul>

Our task is find a policy <em>π </em>: <em>S </em>→ <em>A </em>which our agent will use to take actions in the environment which maximize cumulative reward, i.e.,

<em>T</em>

X <em>t </em><em>γ R</em>(<em>s<sub>t</sub>,a<sub>t</sub>,s<sub>t</sub></em><sub>+1</sub>) (1)

<em>t</em>=0

where <em>γ </em>∈ [0<em>,</em>1] is a discounting factor (used to give more weight to more immediate rewards), <em>s<sub>t </sub></em>is the state at time step <em>t</em>, <em>a<sub>t </sub></em>is the action the agent took at time step <em>t</em>, and <em>s<sub>t</sub></em><sub>+1 </sub>is the state which the environment transitioned to after the agent took the action.

<h2>2.2        Q-Learning</h2>

Q-Learning is a process in which we train some function <em>Q<sub>θ </sub></em>: <em>S </em>× <em>A </em>→ R, parameterized by <em>θ</em>, to learn a mapping from state-action pairs to their <em>Q-value</em>, which is the expected discounted reward for following the following policy <em>π<sub>θ</sub></em>:

<em>π</em>(<em>s<sub>t</sub></em>) = argmax<em>Q<sub>θ</sub></em>(<em>s<sub>t</sub>,a</em>)                                                                                         (2)

<em>a</em>∈<em>A</em>

In words, the function <em>Q<sub>θ </sub></em>will tell us which action will lead to which expected cumulative discounted reward, and our policy <em>π </em>will choose the action <em>a </em>which, ideally, will lead to the maximum such value given the current state <em>s<sub>t</sub></em>.

Originally, Q-Learning was done in a tabular fashion. Here, we would create an |<em>S</em>| × |<em>A</em>| array, our

<em>Q-Table</em>, which would have entries <em>q<sub>i,j </sub></em>where <em>i </em>corresponds to the <em>i</em>th state (the row) and <em>j </em>corresponds to the <em>j</em>th action (the column), so that if <em>s<sub>t </sub></em>is located in the <em>i</em>th row and <em>a<sub>t </sub></em>is the <em>j</em>th column, <em>Q</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>) = <em>q<sub>i,j</sub></em>. We use a value iteration update algorithm to update our Q-values as we explore the environment’s states:

Figure 2: Our update rule for Q-Learning<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

We are using our Q-function recursively to match (following our policy <em>π</em>) in order to calculate the discounted cumulative total reward. We initialize the table with all 0 values, and as we explore the environment

(e.g., take random actions), collect our trajectories, [<em>s</em><sub>0</sub><em>,a</em><sub>0</sub><em>,r</em><sub>0</sub><em>,s</em><sub>1</sub><em>,a</em><sub>2</sub><em>,r</em><sub>2</sub><em>,…,s<sub>T</sub>,a<sub>T</sub>,r<sub>T</sub></em>], and use these values to update our corresponding entries in the Q-table.

During training, the agent will need to explore the environment in order to learn which actions will lead to maximal future discounted rewards. Agent’s often begin exploring the environment by taking random actions at each state. Doing so, however, poses a problem: the agent may not reach states which lead to optimal rewards since they may not take the optimal sequence of actions to get there. In order to account for this, we will slowly encourage the agent to follow it’s policy in order to take actions it believes will lead to maximal rewards. This is called exploitation, and striking a balance between exploration and exploitation is key to properly training an agent to learn to navigate an environment by itself.

To facilitate this, we have our agent follow what is called an -greedy strategy. Here, we introduce a parameter <em> </em>and set it initially to 1. At every training step, we decrease it’s value gradually (e.g., linearly) until we’ve reached some predetermined minimal value (e.g., 0<em>.</em>1). In this case, we are annealing the value of <em> </em>linearly so that it follows a schedule.

During each time step, we have our agent either choose an action according to it’s policy or take a random action by taking a sampling a single value on the uniform distribution over [0<em>,</em>1] and selecting a random action if that sampled value is than <em> </em>otherwise taking an action following the agent’s policy.

<h1>3        Task</h1>

For this project, you will be tasked with both implementing and explaining key components of the Q-learning algorithm. Specifically, we will provide a simple environment and a framework to facilitate training, and you will be responsible for supplying the missing methods of the provided agent classes. For this project, you will be implementing tabular Q-Learning, an approach which utilizes a table of Q-values as the agent’s policy.

<h2>3.1        Environment</h2>

Reinforcement learning environments can take on many different forms, including physical simulations, video games, stock market simulations, etc. The reinforcement learning community (and, specifically, OpenAI) has developed a standard of how such environments should be designed, and the library which facilitates this is OpenAI’s Gym (<a href="https://gym.openai.com/">https://gym.openai.com/</a><a href="https://gym.openai.com/">)</a>.

Figure 3: The initial state of our basic grid-world environment.

The environment we provide is a basic deterministic <em>n </em>× <em>n </em>grid-world environment (the initial state for an 4 × 4 grid-world is shown in Figure 3) where the agent (shown as the green square) has to reach the goal (shown as the yellow square) in the least amount of time steps possible.

The environment’s state space will be described as an <em>n </em>× <em>n </em>matrix with real values on the interval [0<em>,</em>1] to designate different features and their positions. The agent will work within an action space consisting of four actions: <em>up, down, left, right</em>. At each time step, the agent will take one action and move in the direction described by the action. The agent will receive a reward of +1 for moving closer to the goal and −1 for moving away or remaining the same distance from the goal.

<a href="https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26">2 </a><a href="https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26">https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26</a>

<h2>3.2        Code</h2>

This project will not be (relatively) computationally intensive, so most machines will be capable of training the agent quickly. We will provide code templates in the form of Jupyter Notebooks hosted on Google Colab: <a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em>https://colab.research.google.com/drive/1rQjtiSfuZ</em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em><sub>Jd</sub></em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em>rW</em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1">6</a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em>DXtb</em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1">92</a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em>A</em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1">8</a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1"><em>tvUqAyP</em></a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1">1</a><a href="https://colab.research.google.com/drive/1rQjtiSfuZ_J_drW6DXtb92A8tvUqAyP1">.</a>

Google Colab is a free service which hosts Jupyter Notebook instances, giving free access to cloud-based Python environments as well as free GPU/TPU allocation time. This project will be designed to run completely in Google Colab with the intent that you can complete the entire project without having to install anything (although you are welcome to work in whatever environment you want).

We will provide code in the form of a template implementation of the learning agents. You will be tasked with filling in the missing sections to get the implementation working. The expectation will be that the agent is compatible with OpenAI Gym environments. We will also provide a notebook to test your implementation and asses it’s abilities.

Successful implementations will have agents capable of learning to navigate the environment and reach it’s maximum reward on average within a fair amount of epochs.

<h2>3.3        Report</h2>

Along with the code, you will be expected to describe your efforts through a report. The report should contain a section describing your challenge, the Q-learning algorithm, and the environment with which your agent is trained. Each of the key components listed above should be described in it’s own section, including the role it plays in the algorithm, any hyperparameters associated with it and how the hyperparameters affect the agent’s learning, and challenges and details related to your implementation of the component. The report should conclude with results and reflections.

The expectation, here, is that you will be able to describe each of these components at a high-level in your own words. Since the code will be given as a template (and there are a lot of resources online with regards to Q-Learning), we expect the coding portion to be straight-forward. As such, grading will be primarily based on your ability to explain what you have implemented versus the end results.

<h1>4        Deliverables</h1>

You need to submit only one zip-file <em>proj4.zip</em>, that will contain the following:

<ul>

 <li>Jupyter Notebook (<em>ipynb</em>), that will contain your working implementation.</li>

 <li>The report (<em>pdf </em>), which is to be written in L<sup>A</sup>TEX</li>

</ul>

Submit the project on the CSE student server with the following script:

submit_cse474 proj4.zip for undergraduates submit_cse574 proj4.zip for graduates

<h1>5.      Due Date and Time</h1>

The due date is 11:59PM, Dec 4.

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://www.researchgate.net/publication/322424392_Data_Science_in_the_Research_Domain_Criteria_Era_Relevance_of_Machine_Learning_to_the_Study_of_Stress_Pathology_Recovery_and_Resilience">https://www.researchgate.net/publication/322424392_Data_Science_in_the_Research_Domain_Criteria_Era_</a>

<a href="https://www.researchgate.net/publication/322424392_Data_Science_in_the_Research_Domain_Criteria_Era_Relevance_of_Machine_Learning_to_the_Study_of_Stress_Pathology_Recovery_and_Resilience">Relevance_of_Machine_Learning_to_the_Study_of_Stress_Pathology_Recovery_and_Resilience</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://en.wikipedia.org/wiki/Q-learning">https://en.wikipedia.org/wiki/Q-learning</a>