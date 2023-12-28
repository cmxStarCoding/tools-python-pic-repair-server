import os
import pickle


class FileCache:
    def __init__(self, cache_dir='cache/'):
        self.cache_dir = cache_dir
        self.initialize_cache_dir()

    def initialize_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def set(self, key, value):
        cache_path = os.path.join(self.cache_dir, key)
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(value, cache_file)

    def get(self, key):
        cache_path = os.path.join(self.cache_dir, key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                return pickle.load(cache_file)
        return None



