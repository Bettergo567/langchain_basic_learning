from time import sleep

def consumer():
    tx = []
    while True:
        rc = yield tx
        print('[CONSUMER] I have reviced %s...' % rc)
        if rc == None:
            continue
        rc.remove('food')
        print('Eating...')
        sleep(1)
        tx = rc
        print('[CONSUMER] I have eaten the food and sent back %s...' % tx)

def producer(c):
    c.send(None)
    tx = ['plate']
    n = 0
    while n < 3:
        n = n + 1
        tx.append('food')
        print('Producing...')
        sleep(1)
        print('[PRODUCER] I am sending %s to comsumer...' % tx)
        rc = c.send(tx)
        if rc == None:
            continue
        print('[PRODUCER] I have recived %s' % rc)
        tx = rc
    c.close()

c = consumer()
producer(c)