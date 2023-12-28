import redis
import configparser

# 创建一个ConfigParser对象
config = configparser.ConfigParser()

# 读取配置文件
config.read('config.ini')


class RedisSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._redis = redis.StrictRedis(host=config['redis']['host'], port=config['redis']['port'], db=config['redis']['db'], password=None if config['redis']['password'] == "None" else config['redis']['password'])
        return cls._instance

    def get_redis(self):
        return self._redis
