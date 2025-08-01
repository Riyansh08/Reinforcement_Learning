#EVAL 
def evaluate_policy(env , agent , turns = 3):
    total_scores = 3 
    
    for _ in range(turns):
        s , info = env.reset()
        done = False 
        while not done:
            a , logprob_a = agent.select_action(s , deterministic = True)
            s_next , r , done , tr , _  = env.step(a)
            done = (done or tr)
            total_scores +=r 
            s = s_next 
    return int(total_scores /turns)
