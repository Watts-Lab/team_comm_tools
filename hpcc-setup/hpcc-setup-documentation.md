
## HPCC Best Practices and Documentation of Set-Up Steps


### Connecting to HPCC

Logging in from campus:
- open a terminal and run: `ssh USER@hpcc.wharton.upenn.edu`, where "USER" is your Wharton ID

Logging in off campus:
- same step as above, but first it is required to connect to campus VPN
    - Wharton guide to setting up VPN connection [here](https://support.wharton.upenn.edu/help/wharton-vpn)


### Interactive development in head node; qlogin
- the `qlogin` command opens a session on one of the "head nodes" of the cluster, which is where you can develop code and test prior to submitting a job script to the cluster
- entering interactive development session:
    - run: `qlogin -now no`

```
QUESTION: I am still not sure what exactly the `qlogin` command does. 
I understand that it opens a long session, but what am I doing in that session exactly?
```

### Set Up Environment for Development (qlogin)
- instead of setting up configurations, virtual environment, modules, etc. manually, we've created a setup script to launch a Qlogin session with all these steps automatically
- run command from your user directory on HPCC, the default path after connecting via ssh
    - `svn export https://github.com/Watts-Lab/lab-setup/trunk/handbook/HighPerformanceCompute hpcc_setup`
    - this will populate files under the `~/hpcc_setup` directory 

```
ERROR: I am unable to progress past this step because it was looking for a username/password. 
I tried both my Penn and my GitHub passwords, but neither worked.
```


### edit the `env_setup.sh` script:
- line 8: `export USER=...`; 
    - input Penn Key
- line 9: `export AWS_PROFILE=aws-seas-wattslab-acct-...`; 
    - input AWS Credentials Group. Look up your group [here](https://docs.google.com/spreadsheets/d/16TD2y68H6PW07J2AQ0vw5Y9JUR69NbXG/edit#gid=2084621666)
- line 11: `export ENV_NAME=...`;
    - include desired name for default virtual environment


### Run env_setup.sh (configuration for qlogin interactive development env.)
    - execute following from home directory: `chmod 755 env_setup.sh`
    - run `qlogin -now no`
    - execute module_load script: `source env_setup.sh`

### Navigating the HPCC file system
- CSSLab primary shared directory: `/data/projects/CSSLab/`

```
ERROR: I get 'permission denied' here, when I try to cd into this directory.
```


### HPCC Jobs/Job Arrays

### Using Jupyter Lab
- follow General `qlogin` steps above 
- run: `jupyter lab &`
    - Open another terminal window on your local computer and copy / paste the ssh tunnel command from the output from above (below is just an example):
        - example --> `ssh mriv@hpcc.wharton.upenn.edu -f -N -L 47686:hpcc019:47686`
    - Then open a local browser and copy / paste the 127.0.0.1 URL from the output from the 'jupyter lab' command:
        - example --> `http://127.0.0.1:47686/lab?token=41a0db54573edaf50e661b3a88894dbe28c4db9003f`

```
ERROR: It doesn't look like jupyter is installed by default on my hpcc instance. I tried running `jupyter lab &` and got an error, "jupyter: command not found." 

Upon attempting to install jupyterlab, I get the following errors:


pip3 install jupyter-lab
Failed to import the site module
Traceback (most recent call last):
  File "/usr/local/python27/lib/python2.7/site-packages/site.py", line 74, in <module>
    __boot()
  File "/usr/local/python27/lib/python2.7/site-packages/site.py", line 26, in __boot
    import imp # Avoid import loop in Python >= 3.3
  File "/usr/lib64/python3.6/imp.py", line 27, in <module>
    import tokenize
  File "/usr/lib64/python3.6/tokenize.py", line 33, in <module>
    import re
  File "/usr/lib64/python3.6/re.py", line 142, in <module>
    class RegexFlag(enum.IntFlag):
AttributeError: module 'enum' has no attribute 'IntFlag'
```


### Scripting using Vim or Interactive Development Environment (IDE)
- neovim
- git pull/push from local development environment

```
TODO: not sure how to do this yet
```


### Monitoring active jobs
- qstat

### Additional qsub job management resources
https://research-it.wharton.upenn.edu/documentation/job-management/

### Opening Up HPCC Folders Locally
On Mac: Finder > Go > Connect to Server > enter `hpcc.wharton.upenn.edu` > enter UN and PW. This allows you to open your HPCC folders locally (and do more GUI-style drag and drop).
