<h1> Computation noise promotes cognitive resilience </h1>

Authors: Charles Findling and Valentin Wyart

<h3> Summary of the paper </h3>

We show that adding biologically inspired computation noise promotes cognitive resilience in well-controlled behavioral tasks.

<h3> Summary of the code </h3>

This code provides the implementations of the recurrent neural networks (RNNs). You can find two folders:
<ul>
	<li> a `reasoning` folder in which there is the code for the probabilistic reasoning setting. In this setting, we train the RNN on a `associative learning` A task where the agent is taught cue-action associations. Then, without further instructions and feedback, we test it on a `weather prediction` A* task where the agent is presented with sequences of heterogeous cues.
	<li> a `conditioning` folder in which there is the code for the reward-guided learning setting. In this setting, we train the RNN on a bandit task A with fixed reward schedules and then test it on a A* with changing reward schedules.
</ul>
In each folder, you will find: 
<ul>
	<li> a `main.py` file which contains the body of the script (the training and testing functions)
	<li> a `RNNs` folder which contains the code for the RNNs
	<li> two additional folders `REINFORCEtrainings` and `save_models_here` where model information is saved
</ul>

<h3> Code specifications </h3>

python version: Python 3.7.2  
tensorflow version: Tensorflow 1.15.0  
tensorboard version: Tensorboard 2.0.0  

<h3> Code instructions </h3>

In each `main.py` file, there is at the end a python `main` method indicating an example of how to launch the code. There is essentially two arguments, `noise`, `coefficient_list`. The `noise` variable indicates whether we introduce computation noise (`noise=1`) or decision entropy (`noise=0`). Given the `noise` variable, the `coefficient_list` indicates the level of the computation noise or decision entropy.

If you open a terminal and go to one of the two folders (e.g., `reasoning`), and launch the command
```
python main.py
```
Then, the python `main.py` will launch itself. To follow the training visually, launch in another terminal, go the directory and enter the command
```
tensorboard --logdir=REINFORCEtrainings
```

<h3> Questions </h3>
If you have any question, do not hesitate to contact me at charles.findling[at]gmail.com



