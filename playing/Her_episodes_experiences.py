import numpy as np
import random


class her_buffer(object):

    def __init__(self, buffer_size=1000):
        self.memory = []
        self.mem_size = buffer_size

    def clear(self):
        self.memory = []

    def add(self, experience):
        if len(self.memory) + 1 >= self.mem_size:
            self.memory[0:(1 + len(self.memory)) - self.mem_size] = []
        self.memory.extend(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.memory if len(episode) + 1 > trace_length]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 7])
