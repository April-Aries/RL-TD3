from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_test/',
		"update_freq": 4,					# Lab 3: delayed update steps
		"eval_interval": 10,
		"eval_episode": 10,
		"twin_Q_net": True,					# Lab 1: Twin Q-network (True) vs. single Q-network (False)
		"target_policy_smoothing": True,	# Lab 2: Enabling (True) / Disabling (False) target policy smoothing
		"action_noise": "Gaussian"			# Lab 4: Different levels of action noise (OU vs. Gaussian)
	}
	agent = CarRacingTD3Agent(config)
	agent.train()

"""
## Lab Design:

1. Twin Q-network vs. single Q-network
	* config["twin_Q_net"] = True (default), False
2. Enabling / Disabling target policy smoothing
	* config["target_policy_smoothing"] = True (default), False
3. Delayed update steps
	* config["update_freq"] = 1, 4 (default)
4. Different levels of action noise (OU vs. Gaussian)
	* config["action_noise"] = "Gaussian" (default), "OU"

## Logdir

1. Default: config["logdir"] = "log/carRacing/TD3"
2. Lab 1: config["logdir"] = "log/carRacing/Lab1/"
3. Lab 2: config["logdir"] = "log/carRacing/Lab2"
4. Lab 3: config["logdir"] = "log/carRacing/Lab3"
5. Lab 4: config["logdir"] = "log/carRacing/Lab4"
"""