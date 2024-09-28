Code for An LLM-driven Neural Symbolic Framework for Faithful Logical Reasoning

## Setup
* python==3.10


## Experiments
### Datasets
In the folder ``datasets``, we have placed all the datasets required for the experiment.

In each dataset folder, the ``json`` folder places the dataset and the ``npy`` folder places the true answer in .npy file format.


### Main results
To redo our main experiment, you should fill the "LLM_response" and "LLM_response_multi" function in experiments/manual.py. 
LLM_response return an answer to a prompt
LLM_response_multi return a openai choices list


```
# Experimental steps:

1. pip install -r requirements.txt

2. Fill "LLM_response" and "LLM_response_multi"

3. run LLM4logicqa for ReClor and Logicqa

4. run LLM4logicqa for FOLIO, proofwriter and RuleTaker

5. arguments are 
--dataset your dataset path
--type dataset name in "newqa"(logicqa_english) "logicqa"(logicqa chinese) "reclor" "folio" "pw" "ruletaker"
--log logfile
results are the last line in logfiles
```