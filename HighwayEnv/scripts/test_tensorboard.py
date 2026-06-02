import os
from tensorflow.python.summary.summary_iterator import summary_iterator

def list_all_tags(logdir):
    tags = set()
    for file in os.listdir(logdir):
        if file.startswith("events"):
            for e in summary_iterator(os.path.join(logdir, file)):
                for v in e.summary.value:
                    tags.add(v.tag)
    return tags

logdir = 'new1201_dqn/DQN_1'  # Adjust to log directory
all_tags = list_all_tags(logdir)
print("Available Tags:", all_tags)
