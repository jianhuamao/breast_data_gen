import os
class train_log:
    def __init__(self, log_name):
        self.log_name = log_name
        self.dir_path = './log'
        self.log_path = os.path.join(self.dir_path, self.log_name)
    def write_log(self, content):
        with open(self.log_path, 'a') as f:
            f.write(content + '\n')

if __name__ == '__main__':
    test_log = train_log('test_log')
    x = ['line', 'test']
    for i in x:
        test_log.write_log(i)