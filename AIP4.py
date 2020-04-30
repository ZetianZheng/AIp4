#!/usr/bin/env python
# coding: utf-8

# In[3]:


#viterbi algorithm customer journey


# In[108]:


obs = []
ob = [['PRICING'],
       ['NONE'],
       ['PRICING'],
       ['VIDEO','PRICING','BLOG'],
       ['VIDEO','PRICING'],
       ['VIDEO'],
       ['VIDEO','BLOG'],
       ['VIDEO','PRICING'],
       ['BLOG'],
       ['NONE'],
       ['TESTIMONIAL']]
# initialize observation list
for i in range(len(ob)):
    obs.append([s.capitalize() for s in ob[i]])
print(obs)


# In[109]:


states = ('Zero', 'Aware', 'Considering', 'Experiencing', 'Ready', 'Satisfied', 'Lost')
actions = ('Demo', 'Video', 'Testimonial', 'Pricing', 'Blog', 'Payment', 'None')
start_p = {'Zero' : 1, 'Aware' : 0, 'Considering' : 0, 'Experiencing' : 0, 'Ready' : 0, 'Satisfied' : 0, 'Lost' : 0}
trans_p = {
    'Zero' : {'Zero' : 0.6, 'Aware' : 0.4, 'Considering' : 0, 'Experiencing' : 0, 'Ready' : 0, 'Satisfied' : 0, 'Lost' : 0},
    'Aware' : {'Zero' : 0, 'Aware' : 0.49, 'Considering' : 0.3, 'Experiencing' : 0, 'Ready' : 0.01, 'Satisfied' : 0, 'Lost' : 0.2},
    'Considering' : {'Zero' : 0, 'Aware' : 0, 'Considering' : 0.48, 'Experiencing' : 0.2, 'Ready' : 0.02, 'Satisfied' : 0, 'Lost' : 0.3},
    'Experiencing' : {'Zero' : 0, 'Aware' : 0, 'Considering' : 0, 'Experiencing' : 0.4, 'Ready' : 0.3, 'Satisfied' : 0, 'Lost' : 0.3},
    'Ready' : {'Zero' : 0, 'Aware' : 0, 'Considering' : 0, 'Experiencing' : 0, 'Ready' : 0.8, 'Satisfied' : 0, 'Lost' : 0.2},
    'Satisfied': {'Zero' : 0, 'Aware' : 0, 'Considering' : 0, 'Experiencing' : 0, 'Ready' : 0, 'Satisfied' : 1, 'Lost' : 0},
    'Lost' : {'Zero' : 0, 'Aware' : 0, 'Considering' : 0, 'Experiencing' : 0, 'Ready' : 0, 'Satisfied' : 0, 'Lost' : 1}
}
emit_p = {
    'Zero' : {'Demo' : 0.1, 'Video' : 0.01, 'Testimonial' : 0.05, 'Pricing' : 0.3, 'Blog' : 0.5, 'Payment' : 0.0},
    'Aware' : {'Demo' : 0.1, 'Video' : 0.01, 'Testimonial' : 0.15, 'Pricing' : 0.3, 'Blog' : 0.4, 'Payment' : 0.0},
    'Considering' : {'Demo' : 0.2, 'Video' : 0.3, 'Testimonial' : 0.05, 'Pricing' : 0.4, 'Blog' : 0.4, 'Payment' : 0.0},
    'Experiencing' : {'Demo' : 0.4, 'Video' : 0.6, 'Testimonial' : 0.05, 'Pricing' : 0.3, 'Blog' : 0.4, 'Payment' : 0.0},
    'Ready' :  {'Demo' : 0.05, 'Video' : 0.75, 'Testimonial' : 0.35, 'Pricing' : 0.2, 'Blog' : 0.4, 'Payment' : 0.0},
    'Lost' :  {'Demo' : 0.01, 'Video' : 0.01, 'Testimonial' : 0.03, 'Pricing' : 0.05, 'Blog' : 0.2, 'Payment' : 0.0},
    'Satisfied' : {'Demo' : 0.4, 'Video' : 0.4, 'Testimonial' : 0.01, 'Pricing' : 0.05, 'Blog' : 0.5, 'Payment' : 1.0},
}


# In[110]:


# increase 'none' action in emission matrix
for m in emit_p:
    b = 1
    for n in emit_p[m]:
        x = 1 - emit_p[m][n]
        b = round(x*b,2)
    emit_p[m]['None'] = b
print(emit_p)


# In[111]:


# initialization
V = [{}]
for st in  states:
    prob = 1
    for obv in actions:
        if obv in obs[0]:
            prob =  prob * start_p[st] * emit_p[st][obv]
        else:
            prob = prob * start_p[st] * (1 - emit_p[st][obv])
    V[0][st] = {'prob' : prob, 'prev': None}
print(V)


# In[112]:


for t in range(1, len(obs)):
    V.append({})
    for st in states:
        max_tr_prob = V[t-1][states[0]]["prob"] * trans_p[states[0]][st]
        prev_st_selected = states[0]
        for prev_st in states[1:]:
            tr_prob = V[t-1][prev_st]["prob"] * trans_p[prev_st][st]
            if tr_prob > max_tr_prob:
                max_tr_prob = tr_prob
                prev_st_selected = prev_st
        # calculate max probability
        max_prob = 1 
        for obv in actions:
            if obv in obs[t]:
                max_prob = max_prob * max_tr_prob * emit_p[st][obv]
            else:
                max_prob = max_prob * max_tr_prob * (1 - emit_p[st][obv])
        
#         max_prob = max_tr_prob * emit_p[st][obv]
        V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
print(V)


# In[113]:


# def dptable(V):
#     # Print a table of steps from dictionary
#     yield " ".join(("%12d" % i) for i in range(len(V)))
#     for state in V[0]:
#         yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


# In[114]:


# for line in dptable(V):
#     print (line) 


# In[115]:


opt_path = []
max_prob = float("-inf")
previous = None
for st, data in V[-1].items():
    if data['prob'] > max_prob:
        max_prob = data['prob']
        best_st = st
opt_path.append(best_st)
previous = best_st

# follow the back track
for t in range(len(V) - 2, -1, -1):
    opt_path.insert(0, V[t + 1][previous]["prev"])
    previous = V[t + 1][previous]["prev"]
print ('The steps of states are ' + ' '.join(opt_path) + ' with highest probability of %s' % max_prob)


# In[ ]:




